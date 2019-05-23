# PytorchInsight

This is a pytorch lib with state-of-the-art architectures, pretrained models and real-time updated results.

This repository aims to accelarate the advance of Deep Learning Research, make reproducible results and easier for doing researches, and in Pytorch.

## Including Papers (to be updated):
> * SENet: Squeeze-and-excitation Networks <sub>([paper]())</sub>
> * SKNet: Selective Kernel Networks <sub>([paper](https://arxiv.org/pdf/1903.06586.pdf))</sub>
> * SGENet: Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks <sub>([paper]())</sub>



## Trained Models and Performance Table
Single crop validation error on ImageNet-1k (center 224x224/320x320 crop from resized image with shorter side = 256). 

| Model | Top-1 Acc | Top-5 Acc | #P | GFLOPs | Download | log |
|:-:|:-:|:-:|:-:|:-:|:-:|
|ResNet50       |||||[GoogleDrive]()|[log]()|
|SE-ResNet50    ||||||
|SK-ResNet50    ||||||
|BAM-ResNet50   ||||||
|CBAM-ResNet50  ||||||
|SGE-ResNet50   ||||||
|ResNet101      ||||||
|SE-ResNet101   ||||||
|SK-ResNet101   ||||||
|BAM-ResNet101  ||||||
|CBAM-ResNet101 ||||||
|SGE-ResNet101  ||||||




## Citation

If you use Selective Kernel Convolution in your research, please cite the paper:
    
    @inproceedings{li2019selective,
      title={Selective Kernel Networks},
      author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},
      journal={IEEE Conference on Computer Vision and Pattern Recognition},
      year={2019}
    }


