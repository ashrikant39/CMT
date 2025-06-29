import torch, torch.nn.functional as F
import torch.nn as nn
import pdb
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models import VOXEL_ENCODERS, DETECTORS, HEADS
from mmdet3d.models import BACKBONES as BACKBONES_3D
from mmdet.models import BACKBONES as BACKBONES_2D
from mmdet3d.core.bbox import LiDARInstance3DBoxes

@DETECTORS.register_module()
class TransFusion(Base3DDetector):
    
    def __init__(self,
                 voxel_feature_encoder=None,
                 pts_backbone_3d=None,
                 map_to_bev=None,
                 pts_backbone_2d=None,
                 bbox_head=None,
                 init_cfg=None,
                 **kwargs
                 ):
        
        super().__init__(init_cfg=init_cfg)
        
        self.vfe = VOXEL_ENCODERS.build(voxel_feature_encoder)
        self.map_to_bev_module = BACKBONES_2D.build(map_to_bev)
        self.backbone_3d = BACKBONES_3D.build(pts_backbone_3d)
        self.backbone_2d = BACKBONES_2D.build(pts_backbone_2d)
        self.dense_head = HEADS.build(bbox_head)
        
        self.init_weights()
        
    def aug_test(self):
        pass
    
    def simple_test(self):
        pass

    def extract_feat(self, imgs):
        pass
    
    
    def forward_modules(self, points):
        
        batch_dict = dict()
        padded_points = []
        
        for batch_idx, res in enumerate(points):
            padded_res = F.pad(res, (1,0), mode='constant', value=batch_idx)
            padded_points.append(padded_res)
        
        batch_dict['points'] = torch.cat(padded_points, dim=0)
        
        batch_dict = self.dense_head(self.backbone_2d(self.map_to_bev_module(self.backbone_3d(self.vfe(batch_dict)))))
            
        return batch_dict
    

    def _parse_losses(self, losses):
        """
        losses: dict
        
        returns final loss and log info
        """
        loss = losses['loss_trans']
        log_vars = dict()
        
        for key, val in losses:
            log_vars[key] = val.item()
            
        return loss, log_vars
    
    
    def forward_train(
        self,
        points=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs
        ):
        
        #  Points to be rearranged to (batch_idx, x, y, z, i, e)
        # for VFE
        batch_dict = self.forward_modules(points)
        
        outs = batch_dict['outs']
        _, loss_dict = self.dense_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
        
        return loss_dict

    
    
    def forward_test(
        self,
        points=None,
        img_metas=None,
        img=None, **kwargs
        ):
        """
        Return type should be 
        list(dict())
        
        List has length of batch_size
        Dict has keys : ['boxes_3d', 'scores_3d', 'labels_3d']
        
        bbox_results[0]['boxes_3d'] is LiDARInstance3DBoxes, tensor has shape (N, 9) device = cpu
        bbox_results[0]['scores_3d'] is a tensor of shape (N) device = cpu
        bbox_results[0]['labels_3d'] is a tensor of shape (N) device = cpu
        """
        
        batch_dict = self.forward_modules(points)
        outs = batch_dict['outs']
        pred_dicts = self.dense_head.get_bboxes(outs)
        
        predictions = []
        
        for pred in pred_dicts:
            predictions.append(dict(
                boxes_3d = LiDARInstance3DBoxes(pred['pred_boxes'].cpu(), box_dim=9),
                scores_3d = pred['pred_scores'].cpu(),
                labels_3d = pred['pred_labels'].cpu() - 1
            ))
        
        
        return predictions
        