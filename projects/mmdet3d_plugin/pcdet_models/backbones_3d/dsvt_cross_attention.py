import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .dsvt import Stage_Reduction_Block, Stage_ReductionAtt_Block
from math import ceil
from pcdet_models.model_utils.dsvt_utils import PositionEmbeddingLearned
from pcdet_models.model_utils.dsvt_utils import get_window_coors, get_inner_win_inds_cuda, get_pooling_index, get_continous_inds
from .dsvt import _get_activation_fn, DSVT_EncoderLayer, DSVTBlock
from .dsvt_input_layer import DSVTInputLayer
from mmdet3d.models import BACKBONES


class DSVTInputLayerBoxes(nn.Module):
    ''' 
    This class converts the output of vfe to dsvt input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        normalize_pos (bool): Whether to normalize coordinates in position embedding.
    '''
    def __init__(self, model_cfg):
        super().__init__()
        
        self.model_cfg = model_cfg 
        self.sparse_shape = self.model_cfg.sparse_shape # [360, 360, 1]
        self.window_shape = self.model_cfg.window_shape # [[30, 30, 1]]
        self.downsample_stride = self.model_cfg.downsample_stride # []
        self.d_model = self.model_cfg.box_feature_dim # [128]
        self.stage_num = len(self.d_model)

        self.hybrid_factor = self.model_cfg.hybrid_factor # [1, 1, 1]
        self.window_shape = [[self.window_shape[s_id], [self.window_shape[s_id][coord_id] * self.hybrid_factor[coord_id] \
                                                for coord_id in range(3)]] for s_id in range(self.stage_num)] # [[[30,30,1], [30,30,1]]]
        self.shift_list = self.model_cfg.shifts_list # [[[0, 0, 0], [15, 15, 0]]]
        self.normalize_pos = self.model_cfg.normalize_pos # False
            
        self.num_shifts = [2,] * len(self.window_shape) # [2, 2, 2]

        self.sparse_shape_list = [self.sparse_shape] # [[360, 360, 1]]
        # compute sparse shapes for each stage
        for ds_stride in self.downsample_stride:
            last_sparse_shape = self.sparse_shape_list[-1]
            self.sparse_shape_list.append((ceil(last_sparse_shape[0]/ds_stride[0]), ceil(last_sparse_shape[1]/ds_stride[1]), ceil(last_sparse_shape[2]/ds_stride[2])))
        
        # position embedding layers
        # Position Embedding is not global
        # Because there is no shift-info provided to the pos_embed layer.
        self.posembed_layers = nn.ModuleList()
        for i in range(self.stage_num):
            input_dim = 3 if self.sparse_shape_list[i][-1] > 1 else 2 # checking if feature is 2D or 3D
            stage_posembed_layers = nn.ModuleList()
            for s in range(self.num_shifts[i]): # s=0,1
                stage_posembed_layers.append(PositionEmbeddingLearned(input_dim, self.d_model[i]))
            self.posembed_layers.append(stage_posembed_layers)
        
       
    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - pooling_mapping_index_stage{i} (Tensor[int]): Pooling region index used in pooling operation between stage_{i-1} and stage_{i}
                    with shape (N_{i-1}). 
                - pooling_index_in_pool_stage{i} (Tensor[int]): Index inner region with shape (N_{i-1}). Combined with pooling_mapping_index_stage{i}, 
                    we can map each voxel in satge_{i-1} to pooling_preholder_feats_stage{i}, which are input of downsample operation.
                - pooling_preholder_feats_stage{i} (Tensor[int]): Preholder features initial with value 0. 
                    Shape of (N_{i}, downsample_stride[i-1].prob(), d_moel[i-1]), where prob() returns the product of all elements.
                - ...
        '''
        box_feats = batch_dict['box_features']
        box_coors = batch_dict['box_coords'].long()

        box_info = {}
        box_info['box_feats_stage0'] = box_feats.clone()
        box_info['box_coors_stage0'] = box_coors.clone()
        
        for stage_id in range(self.stage_num):
            # window partition of corrsponding stage-map
            box_info = self.window_partition(box_info, stage_id)
            
            for shift_id in range(self.num_shifts[stage_id]):
                box_info[f'box_pos_embed_stage{stage_id}_shift{shift_id}'] = \
                self.get_pos_embed(box_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, shift_id)
        
        return box_info
    
    
    @torch.no_grad()
    def window_partition(self, voxel_info, stage_id):
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(voxel_info[f'box_coors_stage{stage_id}'], 
                                                        self.sparse_shape_list[stage_id], self.window_shape[stage_id][i], i == 1, self.shift_list[stage_id][i])
                                            
            voxel_info[f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win
        
        return voxel_info

    def get_pos_embed(self, coors_in_win, stage_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id]
        embed_layer = self.posembed_layers[stage_id][shift_id]
        
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed


@BACKBONES.register_module()
class DSVTCrossAttention(nn.Module):
    '''Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in set attention for each stage. 
            Length: stage num.
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages. One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.

    '''
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.input_layer = DSVTInputLayer(self.model_cfg.INPUT_LAYER)
        self.box_input_layer = DSVTInputLayerBoxes(self.model_cfg.INPUT_LAYER)
        
        block_name = self.model_cfg.block_name
        set_info = self.model_cfg.set_info
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        self.reduction_type = self.model_cfg.get('reduction_type', 'attention')
        # save GPU memory
        self.use_torch_ckpt = self.model_cfg.get('ues_checkpoint', False)
        self.cross_attention_layers = self.model_cfg.get('cross_attention_layers', [0, 1, 2, 3])
        box_d_model = self.model_cfg.INPUT_LAYER.box_feature_dim[0]
 
        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_name_this_stage = block_name[stage_id]
            block_list=[]
            norm_list=[]
            for i in range(num_blocks_this_stage):
                
                if i in self.cross_attention_layers:
                    block_list.append(
                        DSVTCrossAttentionBlock(dmodel_this_stage, box_d_model, num_head_this_stage, dfeed_this_stage,
                                dropout, activation, batch_first=True)
                    )
                else:                                
                    block_list.append(
                        DSVTBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                                    dropout, activation, batch_first=True)
                    )
                    
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
                
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num-1:
                downsample_window = self.model_cfg.INPUT_LAYER.downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id+1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError

        self.num_shifts = [2] * stage_num
        self.output_shape = self.model_cfg.output_shape
        self.stage_num = stage_num
        self.set_info = set_info
        self.num_point_features = self.model_cfg.conv_out_channel
        
        self._reset_parameters()
        
    
    def get_box_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id]
       
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed
    

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        '''
        voxel_info = self.input_layer(batch_dict)
        box_info = self.box_input_layer(batch_dict)
        
        voxel_feat = voxel_info['voxel_feats_stage0']
        voxel_coors = voxel_info['voxel_coors_stage0']
        box_feat = box_info['box_feats_stage0']
        box_voxel_coors = box_info['box_coors_stage0']
                       
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(self.num_shifts[s])] for b in range(self.set_info[s][1])] for s in range(self.stage_num)]
        pooling_mapping_index = [voxel_info[f'pooling_mapping_index_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_index_in_pool = [voxel_info[f'pooling_index_in_pool_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_preholder_feats = [voxel_info[f'pooling_preholder_feats_stage{s+1}'] for s in range(self.stage_num-1)]
        
        box_pos_embed_list = [[box_info[f'box_pos_embed_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        output = voxel_feat
        
        
        block_id = 0
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(self.set_info[stage_id][-1]):
                
                block = block_layers[i]
                residual = output.clone()
                
                if self.use_torch_ckpt==False:
                    
                    if block.__class__.__name__ == "DSVTBlock":
                        output = block(
                            output,
                            set_voxel_inds_list[stage_id], 
                            set_voxel_masks_list[stage_id], 
                            pos_embed_list[stage_id][i],
                            block_id
                            )
                        
                    else:
                        output = block(
                            output, 
                            set_voxel_inds_list[stage_id], 
                            set_voxel_masks_list[stage_id], 
                            pos_embed_list[stage_id][i],
                            voxel_coors,
                            block_id,
                            box_feat,
                            box_voxel_coors,
                            box_pos_embed_list[stage_id]
                            )
                        
                else:
                    output = checkpoint(block, output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], block_id)

                output = residual_norm_layers[i](output + residual)
                block_id += 1
                
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats[stage_id].type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index[stage_id], pooling_index_in_pool[stage_id]] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)
                
                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output
        batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)


class DSVTCrossAttentionBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, box_d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()
        
        encoder_1 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        encoder_2 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

        # Add cross-attention layer here
        cross_attention_layer_1 = SetCrossAttention(d_model, box_d_model, nhead, dropout, dim_feedforward,
                                        activation, batch_first)
        cross_attention_layer_2 = SetCrossAttention(d_model, box_d_model, nhead, dropout, dim_feedforward,
                                        activation, batch_first)
        self.cross_attn_layer_list = nn.ModuleList([cross_attention_layer_1, cross_attention_layer_2])

    def forward(
            self,
            src,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            voxel_coords,
            block_id,
            box_feature,
            box_voxel_coords,
            box_pos_embed_list
    ):
        
        num_shifts = 2
        output = src
        # TODO: bug to be fixed, mismatch of pos_embed
        for i in range(num_shifts):
            set_id = i
            shift_id = block_id % 2
            pos_embed_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[pos_embed_id]
            layer = self.encoder_list[i]
            box_pos_embed = box_pos_embed_list[i]
            cross_attn_layer = self.cross_attn_layer_list[i]
            
            output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)
            output = cross_attn_layer(output, voxel_coords, box_feature, box_voxel_coords, pos_embed, set_voxel_masks, set_voxel_inds, box_pos_embed)

        return output
    

class SetCrossAttention(nn.Module):

    def __init__(self, d_model, box_d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, kdim=box_d_model, vdim=box_d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.activation = _get_activation_fn(activation)

    def forward(self, 
                src, 
                voxel_coords,
                box_feature, 
                box_voxel_coords,
                pos=None, 
                key_padding_mask=None, 
                voxel_inds=None, 
                box_pos=None,
                onnx_export=False):
        '''
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size).
            voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
            onnx_export (bool): Substitute torch.unique op, which is not supported by tensorrt.
        Returns:
            src (Tensor[float]): Voxel features.
        '''
                
        set_features = src[voxel_inds]
        n_sets, n_voxels_per_set, _ = set_features.shape
    
        if box_pos is not None:
            key = box_feature + box_pos
        
        value = box_feature
            
        key = torch.repeat_interleave(key[None], repeats=n_sets, dim=0)
        value = torch.repeat_interleave(value[None], repeats=n_sets, dim=0)
        
        box_batch_idxs = box_voxel_coords[:,0]
        voxel_batch_idxs = voxel_coords[:,0]
        mask = (voxel_batch_idxs[:,None] != box_batch_idxs[None])[voxel_inds]
        # final_mask = torch.repeat_interleave(torch.logical_or(mask, key_padding_mask[...,None]), repeats=self.nhead, dim=0)
        final_mask = torch.repeat_interleave(mask, repeats=self.nhead, dim=0)
        
        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None
            
        if pos is not None:
            query = set_features + set_pos
            
        src2 = self.cross_attn(query, key, value, attn_mask=final_mask)[0]
        src2 = torch.nan_to_num(src2, nan=0.0)
        
        flatten_inds = voxel_inds.reshape(-1)
        
        if onnx_export:
            src2_placeholder = torch.zeros_like(src).to(src2.dtype)
            src2_placeholder[flatten_inds] = src2.reshape(-1, self.d_model)
            src2 = src2_placeholder
        else:
            unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
            src2 = src2.reshape(-1, self.d_model)[perm]
        
        # FFN layer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

def _get_block_module(name):
    """Return an block module given a string"""
    if name == "DSVTBlock":
        return DSVTBlock
    elif name == 'DSVTCrossAttentionBlock':
        return DSVTCrossAttentionBlock
    else:
        raise RuntimeError(F"This Block not exist.")
    
