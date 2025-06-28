import copy
from functools import reduce

import numpy as np
import torch, math
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, PseudoSampler
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss
from einops import rearrange
import collections

from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, build_bbox_coder)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import NormedLinear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models.utils.clip_sigmoid import clip_sigmoid
from mmdet3d.models import builder
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from functools import reduce
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from projects.mmdet3d_plugin.policynet_models.dense_heads.cmt_head_lyft import SeparateTaskHead
def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        ctx.groups = groups
        ctx.eps = eps
        N, C, L = x.size()
        x = x.view(N, groups, C // groups, L)
        mu = x.mean(2, keepdim=True)
        var = (x - mu).pow(2).mean(2, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y.view(N, C, L) + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        eps = ctx.eps

        N, C, L = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        g = g.view(N, groups, C//groups, L)
        mean_g = g.mean(dim=2, keepdim=True)
        mean_gy = (g * y).mean(dim=2, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx.view(N, C, L), (grad_output * y.view(N, C, L)).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None, None


class GroupLayerNorm1d(nn.Module):

    def __init__(self, channels, groups=1, eps=1e-6):
        super(GroupLayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.groups, self.eps)




class PositionEmbeddingLearned(nn.Module):
    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class MTMDecoder(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear_attn_v = nn.Linear(d_model, d_model)
        self.linear_attn_out = nn.Linear(d_model, d_model)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, attn_map):
        # (B, C, L)
        value = self.linear_attn_v(key.permute(0, 2, 1))
        pre_feature = torch.bmm(attn_map, value)
        pre_feature = self.linear_attn_out(pre_feature)

        # (B, L, C)
        query = query.permute(0, 2, 1)
        query = query + self.dropout1(pre_feature)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm2(query)

        query = query.permute(0, 2, 1)
        return query


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class QTNetHead(nn.Module):
    def __init__(self,
                 num_frames=2,
                 extension=True,
                 pred_weight=0.5,
                 det_weight=0.5,
                 hidden_channel=128,
                 ffn_channel=256,
                 dropout=0.1,
                 activation='relu',
                 hidden_dim=256,
                 train_cfg=None,
                 test_cfg=None,
                 max_diff=[4, 4, 5, 5.5, 3, 0.2, 13, 3, 1, 0.2],
                 common_heads=dict(
                     center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
                 tasks=[
                    dict(num_class=1, class_names=['car']),
                    dict(num_class=2, class_names=['truck','emergency_vehicle']),
                    dict(num_class=2, class_names=['bus','other_vehicle']),
                    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                    dict(num_class=2, class_names=['pedestrian','animal']),
                 ],
                 transformer=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type="FocalLoss",
                     use_sigmoid=True,
                     reduction="mean",
                     gamma=2, alpha=0.25, loss_weight=1.0
                 ),
                 loss_bbox=dict(
                    type="L1Loss",
                    reduction="mean",
                    loss_weight=0.25,
                 ),
                 loss_heatmap=dict(
                     type="GaussianFocalLoss",
                     reduction="mean"
                 ),
                 separate_head=dict(
                     type='SeparateMlpHead', init_bias=-2.19, final_kernel=3),
                 init_cfg=None):
        super(QTNetHead, self).__init__()
        self.num_frames = num_frames
        #self.num_classes = num_classes
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.num_proposals = None
        self.extension = extension
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.fp16_enabled = False

        self.mtm_decoder = MTMDecoder(hidden_channel, ffn_channel, dropout, activation)
        #self.mtm_decoder_reg = MTMDecoder(hidden_channel, ffn_channel, dropout, activation)

        #heads = copy.deepcopy(common_heads)
        #if self.iou_enable:
        #    heads.update(dict(iou=(1, 3)))
        #self.prediction_heads_reg = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)
        #heads = dict(heatmap=(self.num_classes, num_heatmap_convs))
        #self.prediction_heads_cls = FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias)

        # task head
        self.task_heads = nn.ModuleList()
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(cls_logits=(num_cls, 2)))
            separate_head.update(
                in_channels=hidden_dim,
                heads=heads, num_cls=num_cls,
                groups=1
            )
            self.task_heads.append(builder.build_head(separate_head))

        # assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.init_weights()

        self.max_diff = np.asarray(max_diff, dtype=np.float32)
        self.pred_weight = pred_weight
        self.det_weight = det_weight

    def init_weights(self):
        for m in self.mtm_decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        #for m in self.mtm_decoder_reg.parameters():
        #    if m.dim() > 1:
        #        nn.init.xavier_uniform_(m)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def bev2lidar(self, center):
        voxel_size = self.test_cfg['voxel_size']
        pc_range = self.test_cfg['pc_range']
        out_size_factor = self.test_cfg['out_size_factor']
        lidar_coord = center.clone()
        lidar_coord[..., 0] = lidar_coord[..., 0] * voxel_size[0] * out_size_factor + pc_range[0]
        lidar_coord[..., 1] = lidar_coord[..., 1] * voxel_size[1] * out_size_factor + pc_range[1]
        return lidar_coord

    def lidar2bev(self, center, replace=False):
        voxel_size = self.test_cfg['voxel_size']
        pc_range = self.test_cfg['pc_range']
        out_size_factor = self.test_cfg['out_size_factor']
        if replace:
            lidar_coord = center
        else:
            lidar_coord = center.clone()
        lidar_coord[..., 0] = (lidar_coord[..., 0] - pc_range[0]) / voxel_size[0] / out_size_factor
        lidar_coord[..., 1] = (lidar_coord[..., 1] - pc_range[1]) / voxel_size[1] / out_size_factor
        return lidar_coord

    def lidar2bev_boxes(self, boxes):
        voxel_size = self.test_cfg['voxel_size']
        pc_range = self.test_cfg['pc_range']
        out_size_factor = self.test_cfg['out_size_factor']
        targets = torch.zeros([boxes.shape[0], boxes.shape[1], boxes.shape[2] + 1]).to(boxes.device)
        targets[..., 0] = (boxes[..., 0] - pc_range[0]) / (out_size_factor * voxel_size[0])
        targets[..., 1] = (boxes[..., 1] - pc_range[1]) / (out_size_factor * voxel_size[1])
        targets[..., 3] = boxes[..., 3].log()
        targets[..., 4] = boxes[..., 4].log()
        targets[..., 5] = boxes[..., 5].log()
        targets[..., 2] = boxes[..., 2] + boxes[..., 5] * 0.5  # bottom center to gravity center
        targets[..., 6] = torch.sin(boxes[..., 6])
        targets[..., 7] = torch.cos(boxes[..., 6])
        #targets[..., 8] = boxes[..., 7]
        #targets[..., 9] = boxes[..., 8]
        return targets

    def _align_center(self, center, vel, lidar2ego, ego2global, cur_lidar2ego, cur_ego2global):
        B, L, _ = center.size()

        lidar_coord = center.clone()
        lidar_coord[..., 0] = lidar_coord[..., 0] + vel[..., 0] * 0.5
        lidar_coord[..., 1] = lidar_coord[..., 1] + vel[..., 1] * 0.5

        # (B, L, 4) (x, y, z, 1)
        expand_lidar_coord = torch.cat([lidar_coord, center.new_ones((B, L, 2))], dim=-1)
        # TODO BUG: torch.inverse may return NaN.
        #  Use np.linalg.inv to solve the bug.
        lidar2ego = lidar2ego.cpu().numpy()
        ego2global = ego2global.cpu().numpy()
        lidar2ego_inverse = np.linalg.inv(lidar2ego)
        ego2global_inverse = np.linalg.inv(ego2global)
        lidar2ego_inverse = torch.from_numpy(lidar2ego_inverse).to(center.device)
        ego2global_inverse = torch.from_numpy(ego2global_inverse).to(center.device)
        tm = reduce(torch.bmm, [lidar2ego_inverse, ego2global_inverse, cur_ego2global, cur_lidar2ego])
        aligned_lidar_coord = tm.bmm(expand_lidar_coord.permute(0, 2, 1))[:, 0:2].permute(0, 2, 1)
        # (B, H*W, 2)
        aligned_lidar_coord = aligned_lidar_coord.clone()
        return aligned_lidar_coord

    @torch.no_grad()
    def align_center(self, center, vel, index, target_index, img_metas):
        batch_size = len(img_metas)

        cur_lidar2ego_list = list()
        cur_ego2global_list = list()
        prev_lidar2ego_list = list()
        prev_ego2global_list = list()

        for batch in range(batch_size):
            if target_index == 0:
                cur_poses = img_metas[batch]['poses']
            else:
                cur_poses = img_metas[batch]['prev_img_metas'][target_index - 1]['poses']
            cur_lidar2ego_single = torch.from_numpy(cur_poses['lidar2ego']).to(center.device)
            cur_ego2global_single = torch.from_numpy(cur_poses['ego2global']).to(center.device)
            cur_lidar2ego_list.append(cur_lidar2ego_single)
            cur_ego2global_list.append(cur_ego2global_single)

            prev_poses = img_metas[batch]['prev_img_metas'][index - 1]['poses']
            prev_lidar2ego_single = torch.from_numpy(prev_poses['lidar2ego']).to(center.device)
            prev_ego2global_single = torch.from_numpy(prev_poses['ego2global']).to(center.device)
            prev_lidar2ego_list.append(prev_lidar2ego_single)
            prev_ego2global_list.append(prev_ego2global_single)

        cur_lidar2ego = torch.stack(cur_lidar2ego_list, dim=0)
        cur_ego2global = torch.stack(cur_ego2global_list, dim=0)
        prev_lidar2ego = torch.stack(prev_lidar2ego_list, dim=0)
        prev_ego2global = torch.stack(prev_ego2global_list, dim=0)

        aligned_center = self._align_center(center, vel, cur_lidar2ego, cur_ego2global, prev_lidar2ego, prev_ego2global)
        return aligned_center

    def _align_boxes(self, boxes, lidar2ego, ego2global, cur_lidar2ego, cur_ego2global, motion_update=False,
                     forward=True):
        B, L, _ = boxes.size()
        boxes = boxes.clone()
        center = boxes[..., 0:3]
        rot = boxes[..., 6:7]
        #vel = boxes[..., 7:9]
        #extra = boxes[..., 9:]

        if motion_update:
            center[..., 0] = center[..., 0] + vel[..., 0] * 0.5 * 1 if forward else -1
            center[..., 1] = center[..., 1] + vel[..., 1] * 0.5 * 1 if forward else -1

        # (B, L, 4) (x, y, z, 1)
        expand_lidar_coord = torch.cat([center, center.new_ones((B, L, 1))], dim=-1)
        #expand_lidar_vel_coord = torch.cat([vel, vel.new_ones((B, L, 1))], dim=-1)

        rot = rot + torch.atan2(cur_lidar2ego[..., 1, 0], cur_lidar2ego[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)
        rot = rot + torch.atan2(cur_ego2global[..., 1, 0], cur_ego2global[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)
        rot = rot - torch.atan2(ego2global[..., 1, 0], ego2global[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)
        rot = rot - torch.atan2(lidar2ego[..., 1, 0], lidar2ego[..., 0, 0]).unsqueeze(-1).unsqueeze(-1)

        # TODO BUG: torch.inverse may return NaN.
        #  Use np.linalg.inv to solve the bug.
        lidar2ego = lidar2ego.cpu().numpy()
        ego2global = ego2global.cpu().numpy()
        lidar2ego_inverse = np.linalg.inv(lidar2ego)
        ego2global_inverse = np.linalg.inv(ego2global)
        lidar2ego_inverse = torch.from_numpy(lidar2ego_inverse).to(center.device)
        ego2global_inverse = torch.from_numpy(ego2global_inverse).to(center.device)
        tm = reduce(torch.bmm, [lidar2ego_inverse, ego2global_inverse, cur_ego2global, cur_lidar2ego])

        aligned_lidar_coord = tm.bmm(expand_lidar_coord.permute(0, 2, 1))[:, 0:3].permute(0, 2, 1)

        tm = reduce(torch.bmm,
                    [lidar2ego_inverse[:, 0:3, 0:3], ego2global_inverse[:, 0:3, 0:3], cur_ego2global[:, 0:3, 0:3],
                     cur_lidar2ego[:, 0:3, 0:3]])
        #aligned_lidar_vel_coord = tm.bmm(expand_lidar_vel_coord.permute(0, 2, 1))[:, 0:2].permute(0, 2, 1)

        aligned_boxes = torch.cat([aligned_lidar_coord, boxes[..., 3:6], rot], dim=-1)
        return aligned_boxes, tm

    @torch.no_grad()
    def align_boxes(self, boxes, index, target_index, img_metas, motion_update=False, forward=True):
        batch_size = len(img_metas[0])

        cur_lidar2ego_list = list()
        cur_ego2global_list = list()
        prev_lidar2ego_list = list()
        prev_ego2global_list = list()

        for batch in range(batch_size):
            cur_poses = img_metas[target_index][batch]['poses']
            cur_lidar2ego_single = torch.from_numpy(cur_poses['lidar2ego']).to(boxes.device)
            cur_ego2global_single = torch.from_numpy(cur_poses['ego2global']).to(boxes.device)
            cur_lidar2ego_list.append(cur_lidar2ego_single)
            cur_ego2global_list.append(cur_ego2global_single)

            prev_poses = img_metas[index][batch]['poses']
            prev_lidar2ego_single = torch.from_numpy(prev_poses['lidar2ego']).to(boxes.device)
            prev_ego2global_single = torch.from_numpy(prev_poses['ego2global']).to(boxes.device)
            prev_lidar2ego_list.append(prev_lidar2ego_single)
            prev_ego2global_list.append(prev_ego2global_single)

        cur_lidar2ego = torch.stack(cur_lidar2ego_list, dim=0)
        cur_ego2global = torch.stack(cur_ego2global_list, dim=0)
        prev_lidar2ego = torch.stack(prev_lidar2ego_list, dim=0)
        prev_ego2global = torch.stack(prev_ego2global_list, dim=0)

        aligned_boxes, tm = self._align_boxes(boxes, cur_lidar2ego, cur_ego2global, prev_lidar2ego, prev_ego2global,
                                          motion_update, forward)
        return aligned_boxes, tm

    def forward_ffn(self, tokens_cls, tokens_reg, cur_boxes):
        res_layer = self.prediction_heads_cls(tokens_cls)
        res_layer.update(self.prediction_heads_reg(tokens_reg))

        cur_boxes = self.lidar2bev_boxes(cur_boxes)
        cur_center = cur_boxes[..., 0:2]
        cur_height = cur_boxes[..., 2:3]
        cur_dim = cur_boxes[..., 3:6]
        cur_rot = cur_boxes[..., 6:8]
        #cur_vel = cur_boxes[..., 8:10]

        res_layer['center'] = res_layer['center'] + cur_center.permute(0, 2, 1)
        res_layer['height'] = res_layer['height'] + cur_height.permute(0, 2, 1)
        res_layer['dim'] = res_layer['dim'] + cur_dim.permute(0, 2, 1)
        res_layer['rot'] = res_layer['rot'] + cur_rot.permute(0, 2, 1)
        #res_layer['vel'] = res_layer['vel'] + cur_vel.permute(0, 2, 1)

        return res_layer

    @torch.no_grad()
    def mtm_attn(self, prev_center, prev_label, cur_center, cur_label):
        batch_size = prev_center.shape[0]
        # (N, M), detections: N, tracks: M
        N = cur_center.shape[1]
        M = prev_center.shape[1]
        dist = (
            ((prev_center.reshape(batch_size, 1, -1, 2) - cur_center.reshape(batch_size, -1, 1, 2)) ** 2).sum(axis=3))
        dist = torch.sqrt(dist)
        max_diff = torch.from_numpy(self.max_diff).to(prev_label.device)
        max_diff = max_diff[cur_label]
        mask = ((dist > max_diff.reshape(batch_size, N, 1)) + (
                cur_label.reshape(batch_size, N, 1) != prev_label.reshape(batch_size, 1, M))) > 0
        dist = dist + mask * 1e8

        mtm_map = (-1 * dist).softmax(dim=-1)

        return mtm_map

    def forward_temporal_fusion(self, queries, pred_results, img_metas):
        device = queries[0].device
        fused_queries_list_cls = list()
        #fused_queries_list_reg = list()
        fused_boxes_list = list()
        fused_labels_list = list()
        temporal_queries_list_cls = list()
        #temporal_queries_list_reg = list()

        num_frames = self.num_frames  # Total number of frames (e.g., 5)
        num_iterations = num_frames - 2  # Number of iterations (e.g., 3)

        # Mapping of indices:
        # queries[0]: Q(t)
        # queries[1]: Q(t-1)
        # queries[2]: Q(t-2)
        # queries[3]: Q(t-3)
        # queries[4]: Q(t-4)

        for i in range(num_iterations):
            if i == 0:
            # --- Iteration 0 (i = 0) ---
            # Use Q(t-3) and Q(t-4) to predict Q'(t-2)
            # Generate Q^f(t-3) by fusing Q(t-3) and Q(t-4) based on attn_map

                prev_index = num_frames - 1  # Index for Q(t-4), which is 4
                cur_index = num_frames - 2   # Index for Q(t-3), which is 3

                cur_queries = queries[cur_index].to(device)       # Q(t-3)
                cur_boxes = pred_results['boxes'][cur_index].to(device)
                cur_labels = pred_results['labels'][cur_index].long().to(device)

                prev_queries_cls = queries[prev_index].to(device)  # Q(t-4)
                #prev_queries_reg = queries[prev_index].to(device)
                prev_boxes = pred_results['boxes'][prev_index].to(device)
                prev_labels = pred_results['labels'][prev_index].long().to(device)
                #print("i>0")
                #print(prev_queries_cls.shape)
                #print(cur_queries.shape)
                # Align boxes from t-4 to t-3
                aligned_boxes, tm = self.align_boxes(prev_boxes, prev_index, cur_index, img_metas, motion_update=False)
                aligned_prev_center = aligned_boxes[..., 0:2]
                cur_center = cur_boxes[..., 0:2]

                # Compute attention map
                attn_map = self.mtm_attn(aligned_prev_center, prev_labels, cur_center, cur_labels)

                # Decode temporal queries to predict Q'(t-2)
                temporal_queries_cls = self.mtm_decoder(cur_queries, prev_queries_cls, attn_map)
                #temporal_queries_reg = self.mtm_decoder_reg(cur_queries, prev_queries_reg, attn_map)

                # Append Q'(t-2) to temporal_queries_list
                temporal_queries_list_cls.append(temporal_queries_cls)
                #temporal_queries_list_reg.append(temporal_queries_reg)

                # Generate Q^f(t-3) by fusing based on attention map
                if self.extension:
                    prev_mask = torch.max(attn_map, dim=1).values
                    prev_mask = torch.topk(-1 * prev_mask, k=100, dim=-1).indices
                    batch_index = torch.arange(prev_queries_cls.shape[0], dtype=torch.long, device=device).unsqueeze(-1).repeat(1, prev_mask.shape[-1])
                    prev_mask_queries_cls = prev_queries_cls.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                    #prev_mask_queries_reg = prev_queries_reg.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                    prev_mask_boxes = aligned_boxes[batch_index, prev_mask]
                    prev_mask_labels = prev_labels[batch_index, prev_mask]

                    fused_queries_cls = torch.cat([cur_queries, prev_mask_queries_cls], dim=-1)
                    #fused_queries_reg = torch.cat([cur_queries, prev_mask_queries_reg], dim=-1)
                    fused_boxes = torch.cat([cur_boxes, prev_mask_boxes], dim=1)
                    fused_labels = torch.cat([cur_labels, prev_mask_labels], dim=1)
                else:
                    fused_queries_cls = torch.cat([cur_queries, prev_queries_cls], dim=-1)
                    #fused_queries_reg = torch.cat([cur_queries, prev_queries_reg], dim=-1)
                    fused_boxes = torch.cat([cur_boxes, aligned_boxes], dim=1)
                    fused_labels = torch.cat([cur_labels, prev_labels], dim=1)

                # Append fused queries and boxes for Q^f(t-3)
                fused_queries_list_cls.append(fused_queries_cls)
                #fused_queries_list_reg.append(fused_queries_reg)
                fused_boxes_list.append(fused_boxes)
                fused_labels_list.append(fused_labels)

            else:
                # --- Iteration i (i > 0) ---
                # For i = 1: Use Q^f(t-3) and Q'(t-2) to predict Q'(t-1)
                # For i = 2: Use Q^f(t-2) and Q'(t-1) to predict Q'(t)

                # Previous fused queries and data
                prev_queries_cls = fused_queries_list_cls[-1]  # Q^f(t-3) or Q^f(t-2)
                #prev_queries_reg = fused_queries_list_reg[-1]
                prev_boxes = fused_boxes_list[-1]
                prev_labels = fused_labels_list[-1]
                

                # Current queries are the last temporal queries predicted
                cur_queries = temporal_queries_list_cls[-1]  # Q'(t-2) or Q'(t-1)
                cur_boxes = pred_results['boxes'][num_frames - i - 2].to(device)
                cur_labels = pred_results['labels'][num_frames - i - 2].long().to(device)
                #print("i>0")
                #print(prev_queries_cls.shape)
                #print(cur_queries.shape)
                # Align boxes from previous fused to current
                prev_frame_idx = num_frames - i - 1  # Adjusted index for alignment
                cur_frame_idx = num_frames - i - 2

                aligned_boxes, tm = self.align_boxes(prev_boxes, prev_frame_idx, cur_frame_idx, img_metas, motion_update=False)
                aligned_prev_center = aligned_boxes[..., 0:2]
                cur_center = cur_boxes[..., 0:2]

                # Compute attention map
                attn_map = self.mtm_attn(aligned_prev_center, prev_labels, cur_center, cur_labels)

                # Decode temporal queries to predict Q'(t-1) or Q'(t)
                temporal_queries_cls = self.mtm_decoder(cur_queries, prev_queries_cls, attn_map)
                #temporal_queries_reg = self.mtm_decoder_reg(cur_queries, prev_queries_reg, attn_map)

                # Append the new temporal queries
                temporal_queries_list_cls.append(temporal_queries_cls)
                #temporal_queries_list_reg.append(temporal_queries_reg)

                # Generate Q^f(t-2) or Q^f(t-1) by fusing based on attention map
                if self.extension:
                    prev_mask = torch.max(attn_map, dim=1).values
                    prev_mask = torch.topk(-1 * prev_mask, k=100, dim=-1).indices
                    batch_index = torch.arange(prev_queries_cls.shape[0], dtype=torch.long, device=device).unsqueeze(-1).repeat(1, prev_mask.shape[-1])
                    prev_mask_queries_cls = prev_queries_cls.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                    #prev_mask_queries_reg = prev_queries_reg.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                    prev_mask_boxes = aligned_boxes[batch_index, prev_mask]
                    prev_mask_labels = prev_labels[batch_index, prev_mask]

                    fused_queries_cls = torch.cat([cur_queries, prev_mask_queries_cls], dim=-1)
                    #fused_queries_reg = torch.cat([cur_queries, prev_mask_queries_reg], dim=-1)
                    fused_boxes = torch.cat([cur_boxes, prev_mask_boxes], dim=1)
                    fused_labels = torch.cat([cur_labels, prev_mask_labels], dim=1)
                else:
                    fused_queries_cls = torch.cat([prev_queries_cls, temporal_queries_list_cls[-2]], dim=-1)
                    #fused_queries_reg = torch.cat([prev_queries_reg, temporal_queries_list_reg[-2]], dim=-1)
                    fused_boxes = torch.cat([prev_boxes, cur_boxes], dim=1)
                    fused_labels = torch.cat([prev_labels, cur_labels], dim=1)

                # Append fused queries and boxes
                fused_queries_list_cls.append(fused_queries_cls)
                #fused_queries_list_reg.append(fused_queries_reg)
                fused_boxes_list.append(fused_boxes)
                fused_labels_list.append(fused_labels)


        return temporal_queries_list_cls

    def forward_temporal_fusion_org(self, queries, pred_results, img_metas):
        device = queries[0].device
        fused_queries_list_cls = list()
        fused_queries_list_reg = list()
        fused_boxes_list = list()
        fused_labels_list = list()
        temporal_queries_list_cls = list()
        temporal_queries_list_reg = list()

        temporal_queries_list_cls.append(queries[-1])
        temporal_queries_list_reg.append(queries[-1])
        for i in range(self.num_frames - 1):
            prev_index = self.num_frames - i - 1
            cur_index = prev_index - 1
            cur_queries = queries[cur_index].to(device)
            cur_boxes = pred_results['boxes'][cur_index].to(device)
            cur_labels = pred_results['labels'][cur_index].long().to(device)
            if len(fused_queries_list_cls) > 0:
                prev_queries_cls = fused_queries_list_cls[-1]
                prev_queries_reg = fused_queries_list_reg[-1]
                prev_boxes = fused_boxes_list[-1]
                prev_labels = fused_labels_list[-1]
            else:
                prev_queries_cls = prev_queries_reg = queries[prev_index].to(device)
                prev_boxes = pred_results['boxes'][prev_index].to(device)
                prev_labels = pred_results['labels'][prev_index].long().to(device)

            aligned_boxes, tm = self.align_boxes(prev_boxes, prev_index, cur_index, img_metas, motion_update=False)
            aligned_prev_center = aligned_boxes[..., 0:2]
            cur_center = cur_boxes[..., 0:2]

            attn_map = self.mtm_attn(aligned_prev_center, prev_labels, cur_center, cur_labels)
            temporal_queries_cls = self.mtm_decoder_cls(cur_queries, prev_queries_cls, attn_map)
            temporal_queries_reg = self.mtm_decoder_reg(cur_queries, prev_queries_reg, attn_map)
            temporal_queries_list_cls.append(temporal_queries_cls)
            temporal_queries_list_reg.append(temporal_queries_reg)

            if cur_index != 0 and self.extension:
                prev_mask = torch.max(attn_map, dim=1).values
                prev_mask = torch.topk(-1 * prev_mask, k=100, dim=-1).indices
                batch_index = \
                    torch.arange(prev_queries_cls.shape[0], dtype=torch.long).unsqueeze(-1).repeat(1, prev_mask.shape[-1])
                prev_mask_queries_cls = prev_queries_cls.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                prev_mask_queries_reg = prev_queries_reg.permute(0, 2, 1)[batch_index, prev_mask].permute(0, 2, 1)
                prev_mask_boxes = aligned_boxes[batch_index, prev_mask]
                prev_mask_labels = prev_labels[batch_index, prev_mask]
                fused_queries_cls = torch.cat([temporal_queries_cls, prev_mask_queries_cls], dim=-1)
                fused_queries_reg = torch.cat([temporal_queries_reg, prev_mask_queries_reg], dim=-1)
                fused_boxes = torch.cat([cur_boxes, prev_mask_boxes], dim=1)
                fused_labels = torch.cat([cur_labels, prev_mask_labels], dim=1)
            else:
                fused_queries_cls = temporal_queries_cls
                fused_queries_reg = temporal_queries_reg
                fused_boxes = cur_boxes
                fused_labels = cur_labels
            fused_queries_list_cls.append(fused_queries_cls)
            fused_queries_list_reg.append(fused_queries_reg)
            fused_boxes_list.append(fused_boxes)
            fused_labels_list.append(fused_labels)
        return temporal_queries_list_cls, temporal_queries_list_reg

    def forward_single(self,reference, queries, img_metas, pred_results):
        ret_dicts = []
        assert self.num_frames == len(queries)
        self.num_proposals = queries[0].shape[-1]
        device = queries[0].device

        img_metas_list = list()
        img_metas_list.append(img_metas)
        for i in range(1, self.num_frames):
            img_metas_batch = list()
            for j in range(queries[0].shape[0]):
                img_metas_batch.append(img_metas[j]['prev_img_metas'][i - 1])
            img_metas_list.append(img_metas_batch)

        #temporal_queries_cls, temporal_queries_reg = self.forward_temporal_fusion(queries, pred_results, img_metas_list)
        #print(queries[0].shape)
        temporal_queries_cls=self.forward_temporal_fusion(queries, pred_results, img_metas_list)
        flag = 0
        for task_id, task in enumerate(self.task_heads, 0):
            #
            #print("reference")
            #print(reference.shape)
            #print(temporal_queries_cls[-1].unsqueeze(1).shape)
            outs = task(temporal_queries_cls[-1].unsqueeze(1).permute(1, 0, 3, 2))
            
            #print("outs['center']")
            #print(outs['center'].shape)
            center = (outs['center'] + reference[None,:, 0, :, :2]).sigmoid()
            height = (outs['height'] + reference[None,:, 0, :, 2:3]).sigmoid()
            _center, _height = center.new_zeros(center.shape), height.new_zeros(height.shape)
            _center[..., 0:1] = center[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            _center[..., 1:2] = center[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            _height[..., 0:1] = height[..., 0:1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outs['center'] = _center
            outs['height'] = _height
        #cur_boxes = pred_results['boxes'][0].to(device)
        #cur_scores = pred_results['scores'][0].to(device)
        #cur_labels = pred_results['labels'][0].to(device)
        #temporal_res = self.forward_ffn(temporal_queries_cls[-1], temporal_queries_reg[-1], cur_boxes)
        #temporal_res['cur_scores'] = cur_scores
        #temporal_res['cur_labels'] = cur_labels
        ret_dicts.append(outs)
        #res_dict = dict(temporal_results=temporal_res, pred_results=pred_results)
        #return [res_dict]
        return ret_dicts

    def forward(self, reference, queries, img_feats, img_metas, pred_results):
        res = multi_apply(self.forward_single, reference,queries, [img_metas], pred_results)
        assert len(res) == 1, "only support one level features."
        return res
    def _get_targets_single(self, gt_bboxes_3d, gt_labels_3d, pred_bboxes, pred_logits):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            
            gt_bboxes_3d (Tensor):  LiDARInstance3DBoxes(num_gts, 9)
            gt_labels_3d (Tensor): Ground truth class indices (num_gts, )
            pred_bboxes (list[Tensor]): num_tasks x (num_query, 10)
            pred_logits (list[Tensor]): num_tasks x (num_query, task_classes)
        Returns:
            tuple[Tensor]: a tuple containing the following.
                - labels_tasks (list[Tensor]): num_tasks x (num_query, ).
                - label_weights_tasks (list[Tensor]): num_tasks x (num_query, ).
                - bbox_targets_tasks (list[Tensor]): num_tasks x (num_query, 9).
                - bbox_weights_tasks (list[Tensor]): num_tasks x (num_query, 10).
                - pos_inds (list[Tensor]): num_tasks x Sampled positive indices.
                - neg_inds (Tensor): num_tasks x Sampled negative indices.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)
        
        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] - flag2)
            task_boxes.append(torch.cat(task_box, dim=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        
        def task_assign(bbox_pred, logits_pred, gt_bboxes, gt_labels, num_classes):
            num_bboxes = bbox_pred.shape[0]
            assign_results = self.assigner.assign(bbox_pred, logits_pred, gt_bboxes, gt_labels)
            sampling_result = self.sampler.sample(assign_results, bbox_pred, gt_bboxes)
            pos_inds, neg_inds = sampling_result.pos_inds, sampling_result.neg_inds
            # label targets
            labels = gt_bboxes.new_full((num_bboxes, ),
                                    num_classes,
                                    dtype=torch.long)
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights = gt_bboxes.new_ones(num_bboxes)
            # bbox_targets
            code_size = gt_bboxes.shape[1]
            bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
            bbox_weights = torch.zeros_like(bbox_pred)
            bbox_weights[pos_inds] = 1.0
            
            if len(sampling_result.pos_gt_bboxes) > 0:
                bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds

        labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks\
             = multi_apply(task_assign, pred_bboxes, pred_logits, task_boxes, task_classes, self.num_classes)
        
        return labels_tasks, labels_weights_tasks, bbox_targets_tasks, bbox_weights_tasks, pos_inds_tasks, neg_inds_tasks
            
    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            pred_bboxes (list[list[Tensor]]): batch_size x num_task x [num_query, 10].
            pred_logits (list[list[Tensor]]): batch_size x num_task x [num_query, task_classes]
        Returns:
            tuple: a tuple containing the following targets.
                - task_labels_list (list(list[Tensor])): num_tasks x batch_size x (num_query, ).
                - task_labels_weight_list (list[Tensor]): num_tasks x batch_size x (num_query, )
                - task_bbox_targets_list (list[Tensor]): num_tasks x batch_size x (num_query, 9)
                - task_bbox_weights_list (list[Tensor]): num_tasks x batch_size x (num_query, 10)
                - num_total_pos_tasks (list[int]): num_tasks x Number of positive samples
                - num_total_neg_tasks (list[int]): num_tasks x Number of negative samples.
        """
        (labels_list, labels_weight_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_targets_single, gt_bboxes_3d, gt_labels_3d, preds_bboxes, preds_logits
        )
        task_num = len(labels_list[0])
        num_total_pos_tasks, num_total_neg_tasks = [], []
        task_labels_list, task_labels_weight_list, task_bbox_targets_list, \
            task_bbox_weights_list = [], [], [], []

        for task_id in range(task_num):
            num_total_pos_task = sum((inds[task_id].numel() for inds in pos_inds_list))
            num_total_neg_task = sum((inds[task_id].numel() for inds in neg_inds_list))
            num_total_pos_tasks.append(num_total_pos_task)
            num_total_neg_tasks.append(num_total_neg_task)
            task_labels_list.append([labels_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_labels_weight_list.append([labels_weight_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_targets_list.append([bbox_targets_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
            task_bbox_weights_list.append([bbox_weights_list[batch_idx][task_id] for batch_idx in range(len(gt_bboxes_3d))])
        
        return (task_labels_list, task_labels_weight_list, task_bbox_targets_list,
                task_bbox_weights_list, num_total_pos_tasks, num_total_neg_tasks)
        
    def _loss_single_task(self,
                          pred_bboxes,
                          pred_logits,
                          labels_list,
                          labels_weights_list,
                          bbox_targets_list,
                          bbox_weights_list,
                          num_total_pos,
                          num_total_neg):
        """"Compute loss for single task.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            pred_bboxes (Tensor): (batch_size, num_query, 10)
            pred_logits (Tensor): (batch_size, num_query, task_classes)
            labels_list (list[Tensor]): batch_size x (num_query, )
            labels_weights_list (list[Tensor]): batch_size x (num_query, )
            bbox_targets_list(list[Tensor]): batch_size x (num_query, 9)
            bbox_weights_list(list[Tensor]): batch_size x (num_query, 10)
            num_total_pos: int
            num_total_neg: int
        Returns:
            loss_cls
            loss_bbox 
        """
        labels = torch.cat(labels_list, dim=0)
        labels_weights = torch.cat(labels_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)
        
        pred_bboxes_flatten = pred_bboxes.flatten(0, 1)
        pred_logits_flatten = pred_logits.flatten(0, 1)
        
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0.1
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits_flatten, labels, labels_weights, avg_factor=cls_avg_factor
        )

        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]

        loss_bbox = self.loss_bbox(
            pred_bboxes_flatten[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox) 
        return loss_cls, loss_bbox

    def loss_single(self,
                    pred_bboxes,
                    pred_logits,
                    gt_bboxes_3d,
                    gt_labels_3d):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            pred_bboxes (list[Tensor]): num_tasks x [bs, num_query, 10].
            pred_logits (list(Tensor]): num_tasks x [bs, num_query, task_classes]
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        batch_size = pred_bboxes[0].shape[0]
        pred_bboxes_list, pred_logits_list = [], []
        for idx in range(batch_size):
            pred_bboxes_list.append([task_pred_bbox[idx] for task_pred_bbox in pred_bboxes])
            pred_logits_list.append([task_pred_logits[idx] for task_pred_logits in pred_logits])
        cls_reg_targets = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, pred_bboxes_list, pred_logits_list
        )
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._loss_single_task, 
            pred_bboxes,
            pred_logits,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg
        )

        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
    
    def _dn_loss_single_task(self,
                             pred_bboxes,
                             pred_logits,
                             mask_dict):
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        known_labels_raw = mask_dict['known_labels_raw']
        
        pred_logits = pred_logits[(bid, map_known_indice)]
        pred_bboxes = pred_bboxes[(bid, map_known_indice)]
        num_tgt = known_indice.numel()

        # filter task bbox
        task_mask = known_labels_raw != pred_logits.shape[-1]
        task_mask_sum = task_mask.sum()
        
        if task_mask_sum > 0:
            # pred_logits = pred_logits[task_mask]
            # known_labels = known_labels[task_mask]
            pred_bboxes = pred_bboxes[task_mask]
            known_bboxs = known_bboxs[task_mask]

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_tgt * 3.14159 / 6 * self.split * self.split  * self.split
        
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            pred_logits, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_tgt = loss_cls.new_tensor([num_tgt])
        num_tgt = torch.clamp(reduce_mean(num_tgt), min=1).item()

        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = torch.ones_like(pred_bboxes)
        bbox_weights = bbox_weights * bbox_weights.new_tensor(self.train_cfg.code_weights)[None, :]
        # bbox_weights[:, 6:8] = 0
        loss_bbox = self.loss_bbox(
                pred_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_tgt)
 
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        if task_mask_sum == 0:
            # loss_cls = loss_cls * 0.0
            loss_bbox = loss_bbox * 0.0

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    def dn_loss_single(self,
                       pred_bboxes,
                       pred_logits,
                       dn_mask_dict):
        loss_cls_tasks, loss_bbox_tasks = multi_apply(
            self._dn_loss_single_task, pred_bboxes, pred_logits, dn_mask_dict
        )
        return sum(loss_cls_tasks), sum(loss_bbox_tasks)
        
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """"Loss function.
        Args:
            gt_bboxes_3d (list[LiDARInstance3DBoxes]): batch_size * (num_gts, 9)
            gt_labels_3d (list[Tensor]): Ground truth class indices. batch_size * (num_gts, )
            preds_dicts(tuple[list[dict]]): nb_tasks x num_lvl
                center: (num_dec, batch_size, num_query, 2)
                height: (num_dec, batch_size, num_query, 1)
                dim: (num_dec, batch_size, num_query, 3)
                rot: (num_dec, batch_size, num_query, 2)
                cls_logits: (num_dec, batch_size, num_query, task_classes)
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_decoder = preds_dicts[0][0]['center'].shape[0]
        #print(preds_dicts)
        all_pred_bboxes, all_pred_logits = collections.defaultdict(list), collections.defaultdict(list)

        for task_id, preds_dict in enumerate(preds_dicts, 0):
            for dec_id in range(num_decoder):
                pred_bbox = torch.cat(
                    (preds_dict[0]['center'][dec_id], preds_dict[0]['height'][dec_id],
                    preds_dict[0]['dim'][dec_id], preds_dict[0]['rot'][dec_id]),
                    dim=-1
                )
                all_pred_bboxes[dec_id].append(pred_bbox)
                all_pred_logits[dec_id].append(preds_dict[0]['cls_logits'][dec_id])
        all_pred_bboxes = [all_pred_bboxes[idx] for idx in range(num_decoder)]
        all_pred_logits = [all_pred_logits[idx] for idx in range(num_decoder)]
        loss_cls, loss_bbox = multi_apply(
            self.loss_single, all_pred_bboxes, all_pred_logits,
            [gt_bboxes_3d for _ in range(num_decoder)],
            [gt_labels_3d for _ in range(num_decoder)], 
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls[-1]
        loss_dict['loss_bbox'] = loss_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(loss_cls[:-1],
                                           loss_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        #dn_pred_bboxes, dn_pred_logits = collections.defaultdict(list), collections.defaultdict(list)
        #dn_mask_dicts = collections.defaultdict(list)
        #for task_id, preds_dict in enumerate(preds_dicts, 0):
        #    for dec_id in range(num_decoder):
        #        pred_bbox = torch.cat(
        #            (preds_dict[0]['dn_center'][dec_id], preds_dict[0]['dn_height'][dec_id],
        #            preds_dict[0]['dn_dim'][dec_id], preds_dict[0]['dn_rot'][dec_id]),
        #            dim=-1
        #        )
        #        dn_pred_bboxes[dec_id].append(pred_bbox)
        #        dn_pred_logits[dec_id].append(preds_dict[0]['dn_cls_logits'][dec_id])
        #        dn_mask_dicts[dec_id].append(preds_dict[0]['dn_mask_dict'])
        #dn_pred_bboxes = [dn_pred_bboxes[idx] for idx in range(num_decoder)]
        #dn_pred_logits = [dn_pred_logits[idx] for idx in range(num_decoder)]
        #dn_mask_dicts = [dn_mask_dicts[idx] for idx in range(num_decoder)]
        #dn_loss_cls, dn_loss_bbox = multi_apply(
        #    self.dn_loss_single, dn_pred_bboxes, dn_pred_logits, dn_mask_dicts
        #)

        #loss_dict['dn_loss_cls'] = dn_loss_cls[-1]
        #loss_dict['dn_loss_bbox'] = dn_loss_bbox[-1]
        #num_dec_layer = 0
        #for loss_cls_i, loss_bbox_i in zip(dn_loss_cls[:-1],
        #                                   dn_loss_bbox[:-1]):
        #    loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
        #    loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
        #    num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        #print(preds_dicts)
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        #print("decoded")
        #print(preds_dicts)
        num_samples = len(preds_dicts)
        
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_targets_org(self, gt_bboxes_3d, gt_labels_3d, preds_dict, pred_flag=False):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """

        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        pred_flag = [pred_flag for _ in range(len(gt_bboxes_3d))]
        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, pred_flag)

        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        gt_inds = list(res_tuple[7])
        return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, gt_inds

    def get_targets_single_org(self, gt_bboxes_3d, gt_labels_3d, preds_dict, pred_flag=False):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        if not pred_flag:
            num_proposals = preds_dict['center'].shape[-1]

            # get pred boxes, carefully ! donot change the network outputs
            score = copy.deepcopy(preds_dict['heatmap'].detach())
            center = copy.deepcopy(preds_dict['center'].detach())
            height = copy.deepcopy(preds_dict['height'].detach())
            dim = copy.deepcopy(preds_dict['dim'].detach())
            rot = copy.deepcopy(preds_dict['rot'].detach())
            if 'vel' in preds_dict.keys():
                vel = copy.deepcopy(preds_dict['vel'].detach())
            else:
                vel = None
            boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)
            bboxes_tensor = boxes_dict[0]['bboxes']
        else:
            num_proposals = preds_dict['boxes'].shape[1]
            bboxes_tensor = preds_dict['boxes'].squeeze(0)
            scores = preds_dict['scores']
            labels = preds_dict['labels'].long()
            one_hot = F.one_hot(labels, num_classes=self.num_classes)
            score = (one_hot * scores.unsqueeze(-1)).permute(0, 2, 1)

        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        assign_result = self.bbox_assigner.assign(bboxes_tensor, gt_bboxes_tensor, gt_labels_3d, score, self.train_cfg)

        sampling_result = self.bbox_sampler.sample(assign_result, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(score.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(score.device)
        ious = assign_result.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(
            pos_inds.shape[0]), float(mean_iou), assign_result.gt_inds


    def loss_det_org(self, preds_dict, labels, label_weights, num_pos, bbox_weights, bbox_targets, matched_ious, ious,
                 prefix):
        layer_labels = labels.reshape(-1)
        layer_label_weights = label_weights.reshape(-1)
        layer_score = preds_dict['heatmap']
        layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
        layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

        layer_center = preds_dict['center']
        layer_height = preds_dict['height']
        layer_rot = preds_dict['rot']
        layer_dim = preds_dict['dim']
        preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)
        if 'vel' in preds_dict.keys():
            layer_vel = preds_dict['vel']
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)
        code_weights = self.train_cfg.get('code_weights', None)
        layer_bbox_weights = bbox_weights
        layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
        layer_bbox_targets = bbox_targets
        layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

        loss_dict = dict()
        loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
        loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

        if self.iou_enable:
            layer_iou = preds_dict['iou'].squeeze(1)
            layer_loss_iou = self.loss_iou(layer_iou, ious, bbox_weights.max(-1).values,
                                           avg_factor=max(num_pos, 1))
            loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict[f'{prefix}_matched_ious'] = layer_loss_cls.new_tensor(matched_ious)
        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def loss_org(self, gt_bboxes_3d, gt_labels_3d, preds_dicts):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        temporal_res = preds_dicts[0][0]['temporal_results']

        labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, gt_inds = self.get_targets(
            gt_bboxes_3d, gt_labels_3d, [temporal_res])

        loss_dict = dict()
        loss_det = self.loss_det(temporal_res, labels, label_weights, num_pos, bbox_weights, bbox_targets,
                                 matched_ious, ious, prefix='det')

        loss_dict.update(loss_det)
        return loss_dict

    def get_bboxes_org(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        preds_dicts = [[preds_dicts[0][0]['temporal_results']]]
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid()
            if self.iou_enable:
                iou = preds_dict[0]['iou']
                iou = torch.clamp(iou, min=0.0, max=1.0)
                det_weight = iou
            else:
                det_weight = self.det_weight

            batch_score = self.pred_weight * batch_score + det_weight * preds_dict[0]['cur_scores'].unsqueeze(dim=1)

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel,
                                          filter=True)

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]],
                                                          dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res
