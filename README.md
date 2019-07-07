# PytorchInsight

This is a pytorch lib with state-of-the-art architectures, pretrained models and real-time updated results.

This repository aims to accelarate the advance of Deep Learning Research, make reproducible results and easier for doing researches, and in Pytorch.

## Including Papers (to be updated):

#### Attention Models

> * SENet: Squeeze-and-excitation Networks <sub>([paper](https://arxiv.org/pdf/1709.01507.pdf))</sub>
> * SKNet: Selective Kernel Networks <sub>([paper](https://arxiv.org/pdf/1903.06586.pdf))</sub>
> * CBAM: Convolutional Block Attention Module <sub>([paper](https://arxiv.org/pdf/1807.06521.pdf))</sub>
> * GCNet: GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond <sub>([paper](https://arxiv.org/pdf/1904.11492.pdf))</sub>
> * BAM: Bottleneck Attention Module <sub>([paper](https://arxiv.org/pdf/1807.06514v1.pdf))</sub>
> * SGENet: Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks <sub>([paper](https://arxiv.org/pdf/1905.09646.pdf))</sub>
> * SRMNet: SRM: A Style-based Recalibration Module for Convolutional Neural Networks <sub>([paper](https://arxiv.org/pdf/1903.10829.pdf))</sub>

#### Non-Attention Models

> * OctNet: Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution <sub>([paper](https://arxiv.org/pdf/1904.05049.pdf))</sub>
> * imagenet_tricks.py: Bag of Tricks for Image Classification with Convolutional Neural Networks <sub>([paper](https://arxiv.org/pdf/1812.01187.pdf))</sub>



## Trained Models and Performance Table
Single crop validation error on ImageNet-1k (center 224x224 crop from resized image with shorter side = 256). 

||classifiaction training settings for media and large models|
|:-:|:-:|
|Details|RandomResizedCrop, RandomHorizontalFlip; 0.1 init lr, total 100 epochs, decay at every 30 epochs; SGD with naive softmax cross entropy loss, 1e-4 weight decay, 0.9 momentum, 8 gpus, 32 images per gpu|
|Examples| ResNet50 |

||classifiaction training settings for mobile/small models|
|:-:|:-:|
|Details|RandomResizedCrop, RandomHorizontalFlip; 0.4 init lr, total 300 epochs, 5 linear warm up epochs, cosine lr decay; SGD with softmax cross entropy loss and label smoothing 0.1, 4e-5 weight decay on conv weights, 0 weight decay on all other weights, 0.9 momentum, 8 gpus, 128 images per gpu|
|Examples| ShuffleNetV2|

## Typical Training & Testing Tips:

### ShuffleNetV2_1x

You may need to add some models to use in `classification/models/imagenet/__init__.py`, e.g., add 
```python
from .shufflenetv2 import *
```

```python
python -m torch.distributed.launch --nproc_per_node=8 imagenet_mobile.py --cos -a shufflenetv2_1x --data /path/to/imagenet1k/ \
--epochs 300 --wd 4e-5 --gamma 0.1 -c checkpoints/imagenet/shufflenetv2_1x --train-batch 128 --opt-level O0 # Triaing

python -m torch.distributed.launch --nproc_per_node=2 imagenet_mobile.py -a shufflenetv2_1x --data /path/to/imagenet1k/ \
-e --resume ../pretrain/shufflenetv2_1x.pth.tar --test-batch 100 --opt-level O0 # Testing, ~69.6% top-1 Acc
```


### Classification
| Model |#P | GFLOPs | Top-1 Acc | Top-5 Acc | Download | log |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ShuffleNetV2_1x|2.28M|0.151|69.6420|88.7200|[GoogleDrive](https://drive.google.com/open?id=1pRMFnUnDRgXyVo1Gj-MaCb07aeAAhSQo)|[shufflenetv2_1x.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/shufflenetv2_1x.log.txt)|
|ResNet50       |25.56M|4.122|76.3840|92.9080|[BaiduDrive(zuvx)](https://pan.baidu.com/s/1gwvuaqlRT9Sl4rDI9SWn_Q)|[old_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/old_resnet50.log.txt)|
|Oct-ResNet50 (0.125)|||||||
|SRM-ResNet50   |||||||
|SE-ResNet50    |28.09M|4.130|77.1840|93.6720||| 
|SK-ResNet50    |26.15M|4.185|77.5380|93.7000|[BaiduDrive(tfwn)](https://pan.baidu.com/s/1Lx5CNUeRQXOSWjzTlcO2HQ)|[sk_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sk_resnet50.log.txt)|
|BAM-ResNet50   |25.92M|4.205|76.8980|93.4020|[BaiduDrive(z0h3)](https://pan.baidu.com/s/1ijPyAbUNQjlo_BcfDpM9Mg)|[bam_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/bam_resnet50.log.txt)|
|CBAM-ResNet50  |28.09M|4.139|77.6260|93.6600|[BaiduDrive(bram)](https://pan.baidu.com/s/1xSwUg9LiuHfmGGq8nQs4Ug)|[cbam_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/cbam_resnet50.log.txt)|
|GC-ResNet50    |||||||
|SGE-ResNet50   |25.56M|4.127|77.5840|93.6640|[BaiduDrive(gxo9)](https://pan.baidu.com/s/11bb2XBGkTqIoOunaSXOOTg)|[sge_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sge_resnet50.log.txt)|
|ResNet101      |44.55M|7.849|78.2000|93.9060|[BaiduDrive(js5t)](https://pan.baidu.com/s/1gjPo1OQ2DFnJCU1qq39v-g)|[old_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/old_resnet101.log.txt)|
|Oct-ResNet101 (0.125)|||||||
|SRM-ResNet101  |||||||
|SE-ResNet101   |49.33M|7.863|78.4680|94.1020|[BaiduDrive(j2ox)](https://pan.baidu.com/s/1GSvSAlQKFH_tSw1NO88MlA)|[se_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/se_resnet101.log.txt)|
|SK-ResNet101   |45.68M|7.978|78.7920|94.2680|[BaiduDrive(boii)](https://pan.baidu.com/s/1O1giKnrp3MVXZnlrndv8rg)|[sk_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sk_resnet101.log.txt)|
|BAM-ResNet101  |44.91M|7.933|78.2180|94.0180|[BaiduDrive(4bw6)](https://pan.baidu.com/s/19bC9AxHt6lxdJEa2CxE-Zw)|[bam_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/cbam_resnet101.log.txt)|
|CBAM-ResNet101 |49.33M|7.879|78.3540|94.0640|[BaiduDrive(syj3)](https://pan.baidu.com/s/19rcXp5IOOTB0HbxmY-NgUw)|[cbam_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/cbam_resnet101.log.txt)|
|GC-ResNet101   |||||||
|SGE-ResNet101  |44.55M|7.858|78.7980|94.3680|[BaiduDrive(wqn6)](https://pan.baidu.com/s/1X_qZbmC1G2qqdzbIx6C0cQ)|[sge_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sge_resnet101.log.txt)|

### Detection
| Model | #p | GFLOPs | Detector | Neck |  AP50:95 (%) | AP50 (%) | AP75 (%) | Download | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet50      | 23.51M | 88.032  | Faster RCNN  | FPN | 37.5 | 59.1 | 40.6 | [BaiduDrive()]() |
| SGE-ResNet50  | 23.51M | 88.149  | Faster RCNN  | FPN | 38.7 | 60.8 | 41.7 | [BaiduDrive()]() |
| ResNet50      | 23.51M | 88.032  | Mask RCNN    | FPN | 38.6 | 60.0 | 41.9 | [BaiduDrive()]() |
| SGE-ResNet50  | 23.51M | 88.149  | Mask RCNN    | FPN | 39.6 | 61.5 | 42.9 | [BaiduDrive()]() |
| ResNet50      | 23.51M | 88.032  | Cascade RCNN | FPN | 41.1 | 59.3 | 44.8 | [BaiduDrive()]() |
| SGE-ResNet50  | 23.51M | 88.149  | Cascade RCNN | FPN | 42.6 | 61.4 | 46.2 | [BaiduDrive()]() |
| ResNet101     | 42.50M | 167.908 | Faster RCNN  | FPN | 39.4 | 60.7 | 43.0 | [BaiduDrive()]() |
| SGE-ResNet101 | 42.50M | 168.099 | Faster RCNN  | FPN | 41.0 | 63.0 | 44.3 | [BaiduDrive()]() |
| ResNet101     | 42.50M | 167.908 | Mask RCNN    | FPN | 40.4 | 61.6 | 44.2 | [BaiduDrive()]() |
| SGE-ResNet101 | 42.50M | 168.099 | Mask RCNN    | FPN | 42.1 | 63.7 | 46.1 | [BaiduDrive()]() |
| ResNet101     | 42.50M | 167.908 | Cascade RCNN | FPN | 42.6 | 60.9 | 46.4 | [BaiduDrive()]() |
| SGE-ResNet101 | 42.50M | 168.099 | Cascade RCNN | FPN | 44.4 | 63.2 | 48.4 | [BaiduDrive()]() |


| Model | #p | GFLOPs | Detector | Neck | AP small (%) | AP media (%) | AP large (%) | Download | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet50      | 23.51M | 88.032 | RetinaNet | FPN | 19.9 | 39.6 | 48.3 | [BaiduDrive()]() |
| SE-ResNet50   | 26.04M | 88.152 | RetinaNet | FPN | 20.7 | 41.3 | 50.0 | [BaiduDrive()]() |
| SK-ResNet50   | 24.11M | 89.414 | RetinaNet | FPN | 20.2 | 40.9 | 50.4 | [BaiduDrive()]() |
| BAM-ResNet50  | 23.87M | 89.804 | RetinaNet | FPN | 19.6 | 40.1 | 49.9 | [BaiduDrive()]() |
| CBAM-ResNet50 | 26.04M | 88.302 | RetinaNet | FPN | 21.8 | 40.8 | 49.5 | [BaiduDrive()]() |
| SGE-ResNet50  | 23.51M | 88.149 | RetinaNet | FPN | 21.8 | 41.2 | 49.9 | [BaiduDrive()]() |


## Citation

If you use related works in your research, please cite the paper:
    
    @inproceedings{li2019selective,
      title={Selective Kernel Networks},
      author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},
      journal={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2019}
    }

    @inproceedings{li2019spatial,
      title={Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks},
      author={Li, Xiang and Hu, Xiaolin and Yang, Jian},
      journal={Arxiv},
      year={2019}
    }


