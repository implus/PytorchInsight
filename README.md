# PytorchInsight

This is a pytorch lib with state-of-the-art architectures, pretrained models and real-time updated results.

This repository aims to accelarate the advance of Deep Learning Research, make reproducible results and easier for doing researches, and in Pytorch.

## Including Papers (to be updated):
> * SENet: Squeeze-and-excitation Networks <sub>([paper]())</sub>
> * SKNet: Selective Kernel Networks <sub>([paper](https://arxiv.org/pdf/1903.06586.pdf))</sub>
> * SGENet: Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks <sub>([paper]())</sub>



## Trained Models and Performance Table
Single crop validation error on ImageNet-1k (center 224x224/320x320 crop from resized image with shorter side = 256). 

|classifiaction training settings |
|:-:|
|RandomResizedCrop, RandomHorizontalFlip|
|0.1 init lr, total 100 epochs, decay at every 30 epochs|
|sync SGD, naive softmax cross entropy loss, 1e-4 weight decay, 0.9 momentum|
|8 gpus, 32 images per gpu|

### Classification
| Model |#P | GFLOPs | Top-1 Acc | Top-5 Acc | Download | log |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ResNet50       |25.56M|4.122|76.3840|92.9080|||
|SE-ResNet50    |28.09M|4.130|77.1840|93.6720||| 
|SK-ResNet50    |26.15M|4.185|77.5380|93.7000|||
|BAM-ResNet50   |25.92M|4.205|76.8980|93.4020|||
|CBAM-ResNet50  |28.09M|4.128|77.6260|93.6600|||
|SGE-ResNet50   |25.56M|4.127|77.5840|93.6640|[BaiduDrive(gxo9)](https://pan.baidu.com/s/11bb2XBGkTqIoOunaSXOOTg)|[sge_resnet50.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sge_resnet50.log.txt)|
|ResNet101      |44.55M|7.849|78.2000|93.9060|||
|SE-ResNet101   |49.33M|7.863|78.4680|94.2680|||
|SK-ResNet101   |45.68M|7.978|78.7920|94.2680|||
|BAM-ResNet101  |44.91M|7.933|78.2180|94.0180|||
|CBAM-ResNet101 |49.33M|7.861|78.3540|94.0640|||
|SGE-ResNet101  |44.55M|7.858|78.7980|94.3680|[BaiduDrive(wqn6)](https://pan.baidu.com/s/1X_qZbmC1G2qqdzbIx6C0cQ)|[sge_resnet101.log](https://github.com/implus/PytorchInsight/blob/master/pretrain_log/sge_resnet101.log.txt)|

### Detection



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


