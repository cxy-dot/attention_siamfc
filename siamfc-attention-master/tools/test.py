from __future__ import absolute_import

import os
from got10k.experiments import *


from siamfc import TrackerSiamFC

import torch
import numpy as np

from thop import profile
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def evaluate_model_efficiency(model):
    model.eval()
    model = model.cuda()  

    
    template = torch.randn(1, 3, 127, 127).cuda()
    search = torch.randn(1, 3, 255, 255).cuda()

    # ---- 1. 计算 Params 和 FLOPs ----
    macs, params = profile(model, inputs=(template, search))
    print(f"[Efficiency] Params: {params / 1e6:.2f} M | FLOPs: {macs / 1e6:.2f} M")

    # ---- 2. 测试推理速度 ----
    # 预热 GPU
    for _ in range(10):
        _ = model(template, search)

    # 正式计时
    start = time.time()
    for _ in range(100):
        _ = model(template, search)
    end = time.time()

    latency = (end - start) / 100
    fps = 1 / latency
    print(f"[Efficiency] Inference latency: {latency * 1000:.2f} ms | FPS: {fps:.2f}")
##############################################################


if __name__ == '__main__':
    net_path = r'E:\cxy\siamfc-pytorch-master\pretrained_weight_all\siamfc_alexnetv3_FocalLoss_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # 提取 model 并评估效率
    evaluate_model_efficiency(tracker.net)
    print("Using GPU:", torch.cuda.get_device_name(0))

    #root_dir = 'E:\cxy\siamfc-pytorch-master\data\OTB100'
    root_dir = r'E:\cxy\siamfc-pytorch-master\data\vot2016'
    #checkpoint = torch.load(net_path)
    #print("Loaded checkpoint keys:", checkpoint.keys())  
    #tracker.net.load_state_dict(checkpoint['state_dict']) 
    #for name, param in tracker.net.named_parameters():
        #print(name, param.requires_grad)

    #e = ExperimentOTB(root_dir, version=2015)
    e = ExperimentVOT(root_dir, version=2016, experiments='supervised')
    e.run(tracker)
    e.report([tracker.name])


