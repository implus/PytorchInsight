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
> * Understanding the Disharmony between Weight Normalization Family and Weight Decay: e-shifted L2 Regularizer <sub>([to appear]()) 
> * Generalization Bound Regularizer: A Unified Framework for Understanding Weight Decay <sub>([to appear]())
> * mixup: Beyond Empirical Risk Minimization <sub>([paper](https://arxiv.org/pdf/1710.09412.pdf))
> * CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features <sub>([paper](https://arxiv.org/pdf/1905.04899.pdf))

----------------------------------------------------

## Trained Models and Performance Table
Single crop validation error on ImageNet-1k (center 224x224 crop from resized image with shorter side = 256). 

||classifiaction training settings for media and large models|
|:-:|:-:|
|Details|RandomResizedCrop, RandomHorizontalFlip; 0.1 init lr, total 100 epochs, decay at every 30 epochs; SGD with naive softmax cross entropy loss, 1e-4 weight decay, 0.9 momentum, 8 gpus, 32 images per gpu|
|Examples| ResNet50 |
|Note    | The newest code adds one default operation: setting all bias wd = 0, please refer to the theoretical analysis of "Generalization Bound Regularizer: A Unified Framework for Understanding Weight Decay" (to appear), thereby the training accuracy can be slightly boosted|

||classifiaction training settings for mobile/small models|
|:-:|:-:|
|Details|RandomResizedCrop, RandomHorizontalFlip; 0.4 init lr, total 300 epochs, 5 linear warm up epochs, cosine lr decay; SGD with softmax cross entropy loss and label smoothing 0.1, 4e-5 weight decay on conv weights, 0 weight decay on all other weights, 0.9 momentum, 8 gpus, 128 images per gpu|
|Examples| ShuffleNetV2|

## Typical Training & Testing Tips:
### Small Models 
#### ShuffleNetV2_1x
```
python -m torch.distributed.launch --nproc_per_node=8 imagenet_mobile.py --cos -a shufflenetv2_1x --data /path/to/imagenet1k/ \
--epochs 300 --wd 4e-5 --gamma 0.1 -c checkpoints/imagenet/shufflenetv2_1x --train-batch 128 --opt-level O0 --nowd-bn # Triaing

python -m torch.distributed.launch --nproc_per_node=2 imagenet_mobile.py -a shufflenetv2_1x --data /path/to/imagenet1k/ \
-e --resume ../pretrain/shufflenetv2_1x.pth.tar --test-batch 100 --opt-level O0 # Testing, ~69.6% top-1 Acc
```
### Large Models
#### SGE-ResNet
```
python -W ignore imagenet.py -a sge_resnet101 --data /path/to/imagenet1k/ --epochs 100 --schedule 30 60 90 \
--gamma 0.1 -c checkpoints/imagenet/sge_resnet101 --gpu-id 0,1,2,3,4,5,6,7 # Training

python -m torch.distributed.launch --nproc_per_node=8 imagenet_fast.py -a sge_resnet101 --data /path/to/imagenet1k/ \ 
--epochs 100 --schedule 30 60 90 --wd 1e-4 --gamma 0.1 -c checkpoints/imagenet/sge_resnet101 --train-batch 32 \ 
--opt-level O0 --wd-all --label-smoothing 0. --warmup 0 # Training (faster) 
```
```
python -W ignore imagenet.py -a sge_resnet101 --data /path/to/imagenet1k/ --gpu-id 0,1 -e --resume ../pretrain/sge_resnet101.pth.tar \
# Testing ~78.8% top-1 Acc

python -m torch.distributed.launch --nproc_per_node=2 imagenet_fast.py -a sge_resnet101 --data /path/to/imagenet1k/ -e --resume \
../pretrain/sge_resnet101.pth.tar --test-batch 100 --opt-level O0 # Testing (faster) ~78.8% top-1 Acc
```
#### WS-ResNet with e-shifted L2 regularizer, e = 1e-3
```
python -m torch.distributed.launch --nproc_per_node=8 imagenet_fast.py -a ws_resnet50 --data /share1/public/public/imagenet1k/ \
--epochs 100 --schedule 30 60 90 --wd 1e-4 --gamma 0.1 -c checkpoints/imagenet/es1e-3_ws_resnet50 --train-batch 32 \
--opt-level O0 --label-smoothing 0. --warmup 0 --nowd-conv --mineps 1e-3 --el2
```

--------------------------------------------------------
## Results of "SGENet: Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks"
Note the following results (old) do not set the bias wd = 0 for large models

### Classification
| Model |#P | GFLOPs | Top-1 Acc | Top-5 Acc | Download1 | Download2 | log |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ShuffleNetV2_1x|2.28M|0.151|69.6420|88.7200||[GoogleDrive](https://drive.google.com/file/d/1pRMFnUnDRgXyVo1Gj-MaCb07aeAAhSQo/view?usp=sharing)|[shufflenetv2_1x.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/shufflenetv2_1x.log.txt)|
|ResNet50       |25.56M|4.122|76.3840|92.9080|[BaiduDrive(zuvx)](https://pan.baidu.com/s/1gwvuaqlRT9Sl4rDI9SWn_Q)|[GoogleDrive](https://drive.google.com/file/d/1ijUOmyDCSQTU9JaNwOu4_fs1cBXHnHPF/view?usp=sharing)|[old_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/old_resnet50.log.txt)|
|SE-ResNet50    |28.09M|4.130|77.1840|93.6720|||| 
|SK-ResNet50    |26.15M|4.185|77.5380|93.7000|[BaiduDrive(tfwn)](https://pan.baidu.com/s/1Lx5CNUeRQXOSWjzTlcO2HQ)|[GoogleDrive](https://drive.google.com/file/d/1DGYWPeKc7dyJ9i-zPJcPPa2engExPOnJ/view?usp=sharing)|[sk_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sk_resnet50.log.txt)|
|BAM-ResNet50   |25.92M|4.205|76.8980|93.4020|[BaiduDrive(z0h3)](https://pan.baidu.com/s/1ijPyAbUNQjlo_BcfDpM9Mg)|[GoogleDrive](https://drive.google.com/file/d/1K5iAUAIF_yRyC2pIiA65F8Ig0x4NzOqk/view?usp=sharing)|[bam_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/bam_resnet50.log.txt)|
|CBAM-ResNet50  |28.09M|4.139|77.6260|93.6600|[BaiduDrive(bram)](https://pan.baidu.com/s/1xSwUg9LiuHfmGGq8nQs4Ug)|[GoogleDrive](https://drive.google.com/open?id=1Q5gIKPARrZzDbCPZHpuj9tqXs06c2YZN)|[cbam_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/cbam_resnet50.log.txt)|
|SGE-ResNet50   |25.56M|4.127|77.5840|93.6640|[BaiduDrive(gxo9)](https://pan.baidu.com/s/11bb2XBGkTqIoOunaSXOOTg)|[GoogleDrive](https://drive.google.com/open?id=13HPCjrEle6aFbiCo8Afkr2jJssdNwdRn)|[sge_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sge_resnet50.log.txt)|
|ResNet101      |44.55M|7.849|78.2000|93.9060|[BaiduDrive(js5t)](https://pan.baidu.com/s/1gjPo1OQ2DFnJCU1qq39v-g)|[GoogleDrive](https://drive.google.com/file/d/1125qwL4psGqJWrPDtSoxfBRLPMAnRzx4/view?usp=sharing)|[old_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/old_resnet101.log.txt)|
|SE-ResNet101   |49.33M|7.863|78.4680|94.1020|[BaiduDrive(j2ox)](https://pan.baidu.com/s/1GSvSAlQKFH_tSw1NO88MlA)|[GoogleDrive](https://drive.google.com/file/d/1MOGkkqs6v_LCgO6baGDmcFYbuOkwZjK9/view?usp=sharing)|[se_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/se_resnet101.log.txt)|
|SK-ResNet101   |45.68M|7.978|78.7920|94.2680|[BaiduDrive(boii)](https://pan.baidu.com/s/1O1giKnrp3MVXZnlrndv8rg)|[GoogleDrive](https://drive.google.com/file/d/1WB7HXx-cvUIxFRe-M61XZIzUN0a3nsbF/view?usp=sharing)|[sk_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sk_resnet101.log.txt)|
|BAM-ResNet101  |44.91M|7.933|78.2180|94.0180|[BaiduDrive(4bw6)](https://pan.baidu.com/s/19bC9AxHt6lxdJEa2CxE-Zw)|[GoogleDrive](https://drive.google.com/open?id=15EUQ6aAoXzPm1YeAH4ZqnF3orEr0dB8f)|[bam_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/cbam_resnet101.log.txt)|
|CBAM-ResNet101 |49.33M|7.879|78.3540|94.0640|[BaiduDrive(syj3)](https://pan.baidu.com/s/19rcXp5IOOTB0HbxmY-NgUw)|[GoogleDrive](https://drive.google.com/file/d/1UHLt3C59M1fRta6i9iLsj-RvIbKusgQN/view?usp=sharing)|[cbam_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/cbam_resnet101.log.txt)|
|SGE-ResNet101  |44.55M|7.858|78.7980|94.3680|[BaiduDrive(wqn6)](https://pan.baidu.com/s/1X_qZbmC1G2qqdzbIx6C0cQ)|[GoogleDrive](https://drive.google.com/file/d/1ihu0NVvVJZEv0zj49izapn4V0FhwxCh6/view?usp=sharing)|[sge_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sge_resnet101.log.txt)|

### Detection
| Model | #p | GFLOPs | Detector | Neck |  AP50:95 (%) | AP50 (%) | AP75 (%) | Download | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet50      | 23.51M | 88.0  | Faster RCNN  | FPN | 37.5 | 59.1 | 40.6 | [GoogleDrive](https://drive.google.com/file/d/1IN3Wr_MyrOVm4Kgyx-Fr-ScdaP6vbWaP/view?usp=sharing) |
| SGE-ResNet50  | 23.51M | 88.1  | Faster RCNN  | FPN | 38.7 | 60.8 | 41.7 | [GoogleDrive](https://drive.google.com/file/d/1XFxix0YZ40viyIyE5KsEWBcS_I8QrWzM/view?usp=sharing) |
| ResNet50      | 23.51M | 88.0  | Mask RCNN    | FPN | 38.6 | 60.0 | 41.9 | [GoogleDrive](https://drive.google.com/file/d/1P9Vu-AOC0EbFK3sJSo45rCvNGMdrF7AZ/view?usp=sharing) |
| SGE-ResNet50  | 23.51M | 88.1  | Mask RCNN    | FPN | 39.6 | 61.5 | 42.9 | [GoogleDrive](https://drive.google.com/file/d/1obT-MQ_eIxfDAcy6a4xs1DKdjYKCpl1u/view?usp=sharing) |
| ResNet50      | 23.51M | 88.0  | Cascade RCNN | FPN | 41.1 | 59.3 | 44.8 | [GoogleDrive](https://drive.google.com/file/d/1aYHeV4O48z9V4Io6Xq1L4-L3_3WfRsGy/view?usp=sharing) |
| SGE-ResNet50  | 23.51M | 88.1  | Cascade RCNN | FPN | 42.6 | 61.4 | 46.2 | [GoogleDrive](https://drive.google.com/file/d/1Bmxlg5qv9b3_Z2PjQ4bTjSPWQ5PvLxxr/view?usp=sharing) |
| ResNet101     | 42.50M | 167.9 | Faster RCNN  | FPN | 39.4 | 60.7 | 43.0 | [GoogleDrive](https://drive.google.com/file/d/1R4RGAp0PlZ8eQr6KNk7tvP8XLvKuYI-p/view?usp=sharing) |
| SE-ResNet101  | 47.28M | 168.3 | Faster RCNN  | FPN | 40.4 | 61.9 | 44.2 | [GoogleDrive](https://drive.google.com/file/d/14BHVJ_grTJXUvKCUsH9PwR-n5U7pussJ/view?usp=sharing) |
| SGE-ResNet101 | 42.50M | 168.1 | Faster RCNN  | FPN | 41.0 | 63.0 | 44.3 | [GoogleDrive](https://drive.google.com/file/d/1TY-n2bKvOIXQ4sj8VHwzQn0cCYUDHA13/view?usp=sharing) |
| ResNet101     | 42.50M | 167.9 | Mask RCNN    | FPN | 40.4 | 61.6 | 44.2 | [GoogleDrive](https://drive.google.com/file/d/1bSXAVo25dUq56BU9rmQgBK7cpx_Cn2lY/view?usp=sharing) |
| SE-ResNet101  | 47.28M | 168.3 | Mask RCNN    | FPN | 41.5 | 63.0 | 45.3 | [GoogleDrive](https://drive.google.com/file/d/1BV4WGgmUjU5oDfiH46FH-7iunkaEyjNv/view?usp=sharing) |
| SGE-ResNet101 | 42.50M | 168.1 | Mask RCNN    | FPN | 42.1 | 63.7 | 46.1 | [GoogleDrive](https://drive.google.com/file/d/1sGMhVJcsm922c-pjbny12kwVRf0v0Hfa/view?usp=sharing) |
| ResNet101     | 42.50M | 167.9 | Cascade RCNN | FPN | 42.6 | 60.9 | 46.4 | [GoogleDrive](https://drive.google.com/file/d/1_scOlE4MWAZWdSk3vVCYDTpy9OEVEsvN/view?usp=sharing) |
| SE-ResNet101  | 47.28M | 168.3 | Cascade RCNN | FPN | 43.4 | 62.2 | 47.2 | [GoogleDrive](https://drive.google.com/file/d/1rKHXxSgJmCAG9oO3V_8oUBgo0WKdOaXA/view?usp=sharing) |
| SGE-ResNet101 | 42.50M | 168.1 | Cascade RCNN | FPN | 44.4 | 63.2 | 48.4 | [GoogleDrive](https://drive.google.com/file/d/1rXII_efJwI7suttG0q6HojQ_aeIhiiYX/view?usp=sharing) |

--------------------------------------------------------
## Results of "Understanding the Disharmony between Weight Normalization Family and Weight Decay: e-shifted L2 Regularizer"
Note that the following models are with bias wd = 0.

### Classification
|Model      | Top-1 | Download |
|:-:|:-:|:-:|
|WS-ResNet50           | 76.74 | [GoogleDrive](https://drive.google.com/file/d/1AeZc_4o5XA8a3av8M3NAOipsy_sV_tgH/view?usp=sharing) |
|WS-ResNet50(e = 1e-3) | 76.86 | [GoogleDrive](https://drive.google.com/file/d/18U_PzzWhOL4GPB7jF36XlCltbAD7Qdcx/view?usp=sharing) |
|WS-ResNet101          | 78.07 | [GoogleDrive](https://drive.google.com/file/d/1LKHq5gxhT0S6L1OFXlw6azWEsEtobxlF/view?usp=sharing) | 
|WS-ResNet101(e = 1e-6)| 78.29 | [GoogleDrive](https://drive.google.com/file/d/12WQ3oCRCGvM9eU9YAbr6jk_2NcpAsyhS/view?usp=sharing) | 
|WS-ResNeXt50(e = 1e-3) |77.88      |[GoogleDrive](https://drive.google.com/file/d/18U_PzzWhOL4GPB7jF36XlCltbAD7Qdcx/view?usp=sharing)| 
|WS-ResNeXt101(e = 1e-3)|78.80      |[GoogleDrive](https://drive.google.com/file/d/14YxWswfC8nyxH34AGQFOT-csSP4aI5vT/view?usp=sharing)|
|WS-DenseNet201(e = 1e-8)  | 77.59  |[GoogleDrive](https://drive.google.com/file/d/1I6XEYBLO-488vBUoyexXEmEzTDnv9wuf/view?usp=sharing)|
|WS-ShuffleNetV1(e = 1e-8) | 68.09  |[GoogleDrive](https://drive.google.com/file/d/1hU8_SJNgFk9uNr8cNCGqDFgkBvm4RdjO/view?usp=sharing)|
|WS-ShuffleNetV2(e = 1e-8) | 69.70  |[GoogleDrive](https://drive.google.com/file/d/1Oc04IvP9JTFM8yDnlbmB5wnugr_3Cd0I/view?usp=sharing)|
|WS-MobileNetV1(e = 1e-6)  | 73.60  |[GoogleDrive](https://drive.google.com/file/d/17oAS8W2Mr83qhgI-gTRG1H6WJGMQdFMB/view?usp=sharing)|

--------------------------------------------------------
## Results of "Generalization Bound Regularizer: A Unified Framework for Understanding Weight Decay"

### To appear

--------------------------------------------------------
## Citation

If you find our related works useful in your research, please consider citing the paper:
    
    @inproceedings{li2019selective,
      title={Selective Kernel Networks},
      author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},
      journal={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2019}
    }

    @inproceedings{li2019spatial,
      title={Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks},
      author={Li, Xiang and Hu, Xiaolin and Xia, Yan and Yang, Jian},
      journal={arXiv preprint arXiv:1905.09646},
      year={2019}
    }

    @inproceedings{li2019understanding,
      title={Understanding the Disharmony between Weight Normalization Family and Weight Decay: e-shifted L2 Regularizer},
      author={Li, Xiang and Chen, Shuo and Yang, Jian},
      journal={arXiv preprint arXiv:},
      year={2019}
    }

    @inproceedings{li2019generalization,
      title={Generalization Bound Regularizer: A Unified Framework for Understanding Weight Decay},
      author={Li, Xiang and Chen, Shuo and Gong, Chen and Xia, Yan and Yang, Jian},
      journal={arXiv preprint arXiv:},
      year={2019}
    }






