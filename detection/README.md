## Quick Examples

### Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.

### Train with 8 gpus
```shell
dist_train.sh local_configs/cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50.py 8
```

### Valid
```shell
python tools/test.py local_configs/cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50.py work_dirs/cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50/epoch_20.pth --gpus 2 --out logs/val.cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50.results.pkl --eval bbox > logs/val.cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50
```


## Introduction

The master branch works with **PyTorch 1.0**. 

Our project is based on [mmdetection](https://github.com/open-mmlab/mmdetection), which is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).


## Inference with pretrained models

### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize detection results

We allow to run one or multiple processes on each GPU, e.g. 8 processes on 8 GPU
or 16 processes on 8 GPU. When the GPU workload is not very heavy for a single
process, running multiple processes will accelerate the testing, which is specified
with the argument `--proc_per_gpu <PROCESS_NUM>`.


To test a dataset and save the results.

```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE>
```

To perform evaluation after testing, add `--eval <EVAL_TYPES>`. Supported types are:
`[proposal_fast, proposal, bbox, segm, keypoints]`.
`proposal_fast` denotes evaluating proposal recalls with our own implementation,
others denote evaluating the corresponding metric with the official coco api.

For example, to evaluate Mask R-CNN with 8 GPUs and save the result as `results.pkl`.

```shell
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py <CHECKPOINT_FILE> --gpus 8 --out results.pkl --eval bbox segm
```

It is also convenient to visualize the results during testing by adding an argument `--show`.

```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --show
```

### Test image(s)

We provide some high-level apis (experimental) to test an image.

```python
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
img = mmcv.imread('test.jpg')
result = inference_detector(model, img, cfg)
show_result(img, result)

# test a list of images
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result)
```


## Train a model

mmdetection implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

### Distributed training (Single or Multiples machines)

mmdetection potentially supports multiple launch methods, e.g., PyTorch’s built-in launch utility, slurm and MPI.

We provide a training script using the launch utility provided by PyTorch.

```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be replaced.

Expected results in WORK_DIR:

- log file
- saved checkpoints (every k epochs, defaults=1)
- a symbol link to the latest checkpoint

**Important**: The default learning rate is for 8 GPUs. If you use less or more than 8 GPUs, you need to set the learning rate proportional to the GPU num. E.g., modify lr to 0.01 for 4 GPUs or 0.04 for 16 GPUs.

### Non-distributed training

Please refer to `tools/train.py` for non-distributed training, which is not recommended
and left for debugging. Even on a single machine, distributed training is preferred.

