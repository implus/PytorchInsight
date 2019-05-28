python -W ignore imagenet.py -a old_resnet50 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/old_resnet50.pth.tar 

python -W ignore imagenet.py -a old_resnet101 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/old_resnet101.pth.tar 

python -W ignore imagenet.py -a sk_resnet50 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/sk_resnet50.pth.tar

python -W ignore imagenet.py -a sk_resnet101 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/sk_resnet101.pth.tar

python -W ignore imagenet.py -a cbam_resnet50 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/cbam_resnet50.pth.tar

python -W ignore imagenet.py -a cbam_resnet101 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/cbam_resnet101.pth.tar

python -W ignore imagenet.py -a bam_resnet50 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/bam_resnet50.pth.tar

python -W ignore imagenet.py -a bam_resnet101 --data /share1/classification_data/imagenet1k/ --gpu-id 0 -e --resume ../pretrain/bam_resnet101.pth.tar
