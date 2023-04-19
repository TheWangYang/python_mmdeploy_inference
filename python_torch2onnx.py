from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'images/demo.jpg'
work_dir = 'work_dir/onnx/fcos'
save_file = 'fcos_end2end.onnx'
deploy_cfg = 'mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/fcos_r50_caffe_fpn_gn-head_1x_spdfk_coco_3000_2000.py'
model_checkpoint = 'checkpoints/latest.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)