import torch.nn as nn
from mmdet3d.models import VOXEL_ENCODERS

@VOXEL_ENCODERS.register_module()
class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError
