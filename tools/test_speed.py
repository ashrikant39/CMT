# Copyright (c) 2023 megvii-model. All Rights Reserved.

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
import spconv.pytorch as spconv
from fvcore.nn import FlopCountAnalysis, flop_count_table
from fvcore.nn.jit_handles import get_shape

import os
import time
import importlib
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2
import pickle
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F 
import matplotlib


class BatchWrapper(nn.Module):
    
    def __init__(self, model, batch_dict):
        super().__init__()
        self.model = model
        self.batch_dict = batch_dict
        
    def forward(self, dummy_input):
        return self.model(self.batch_dict)
    

# Define custom FLOP counting function
def count_sparseconv3d(m, inputs, outputs):
    """
    Estimate FLOPs for SparseConv3d:
    FLOPs = num_active_sites * kernel_volume * in_channels * out_channels
    """
    input_tensor = inputs[0]  # This is SparseConvTensor
    num_active_sites = input_tensor.features.shape[0]
    
    kx, ky, kz = m.kernel_size
    kernel_volume = kx * ky * kz
    in_channels = m.in_channels
    out_channels = m.out_channels
    
    total_flops = num_active_sites * kernel_volume * in_channels * out_channels
    return total_flops


class Wrapper:

    def __init__(self,
                 cfg,
                 checkpoint=None) -> None:
        self.cfg = Config.fromfile(cfg)
        self.save_dir = './tmp'
        self.init()
        self.model = self._build_model(checkpoint)
        self.dataset = self._build_dataset()

    def init(self):
        self.cfg.model.pretrained = None
        self.cfg.data.test.test_mode = True
        plugin_dir = self.cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)

    def _build_model(self, checkpoint=None):
        model = build_detector(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        if checkpoint:
            load_checkpoint(model, checkpoint, map_location='cpu')
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        return model
    
    def _build_dataset(self):
        dataset = build_dataset(self.cfg.data.val)
        return dataset

    def test_speed(self, num_iters=100, amp=False):
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        loader = iter(data_loader)        
        total_time = 0
        
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():
                for _ in range(num_iters):
                    data = next(loader)
                    t1 = time.time()
                    self.model(**data, return_loss=False)
                    total_time += time.time() - t1
        
        print(f'Average time: {total_time / num_iters}')
    
    
    def test_flops(self, num_iters=100, amp=False):
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        loader = iter(data_loader)        
        
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():
                for _ in range(num_iters):
                    data = next(loader)
                    wrapper_model = BatchWrapper(self.model, data)
                    fvcore_custom_ops = {spconv.SparseConv3d: count_sparseconv3d}
                    dummy_input = torch.zeros(1)
                    flops = FlopCountAnalysis(wrapper_model, (dummy_input,), custom_ops=fvcore_custom_ops)
                    print(f"\nfvcore Results:")
                    print(flop_count_table(flops, max_depth=5))


if __name__ == '__main__':
    wrapper = Wrapper(
        cfg='projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py',
    )
    wrapper.test_flops(amp=False)
    