from mmdeploy.apis import inference_model


deploy_cfg = 'mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/fcos_r50_caffe_fpn_gn-head_1x_spdfk_coco_3000_2000.py'
backend_files = ['work_dir/onnx/fcos/fcos_end2end.onnx']
img = ['images/1_1.jpg']
device = 'cpu'

result = inference_model(model_cfg, deploy_cfg, backend_files, img, device)

print("result: {}".format(result))



