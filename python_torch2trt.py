from mmdeploy.apis import torch2onnx
from mmdeploy.apis.tensorrt import onnx2tensorrt
from mmdeploy.backend.sdk.export_info import export2SDK
import os

img = 'images/demo.jpg'
work_dir = 'work_dir/trt/fcos'
save_file = 'fcos_end2end.onnx'
deploy_cfg = 'mmdeploy/configs/mmdet/detection/detection_tensorrt_static-800x1344.py'
model_cfg = 'configs/fcos_r50_caffe_fpn_gn-head_1x_spdfk_coco_3000_2000.py'
model_checkpoint = 'checkpoints/latest.pth'
device = 'cpu'

# 1. convert model to IR(onnx)
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. convert IR to tensorrt
onnx_model = os.path.join(work_dir, save_file)
save_file = 'fcos_end2end.engine'
model_id = 0
device = 'cuda'
onnx2tensorrt(work_dir, save_file, model_id, deploy_cfg, onnx_model, device)

# 3. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)