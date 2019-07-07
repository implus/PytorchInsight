# ShuffleNetV2 
python -m torch.distributed.launch --nproc_per_node=8 imagenet_mobile.py --cos -a shufflenetv2_1x --data /share1/classification_data/imagenet1k/ --epochs 300 --wd 4e-5 --gamma 0.1 -c checkpoints/imagenet/shufflenetv2_1x --train-batch 128 --opt-level O0




----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------00
# old
python -W ignore imagenet_fast.py -a gl128gbn_resnet101 --data /share1/classification_data/imagenet1k/ --epochs 100 --schedule 30 60 90 --gamma 0.1 -c checkpoints/imagenet/gl128gbn_resnet101 --gpu-id 0,1,2,3,4,5,6,7

# TR
python -m torch.distributed.launch --nproc_per_node=8 imagenet_trickiter.py --cos -a gl4gbn_shufflenetv2_1x --data /share1/classification_data/imagenet1k/ --epochs 300 --wd 4e-5 --gamma 0.1 -c checkpoints/imagenet/TR0_gl4gbn_shufflenetv2_1x --train-batch 128 --opt-level O0

# TM
python -m torch.distributed.launch --nproc_per_node=8 imagenet_tricks.py --cos -a shufflenetv2_1x --data /share1/classification_data/imagenet1k/ --epochs 300 --wd 4e-5 --gamma 0.1 -c checkpoints/imagenet/TM0_shufflenetv2_1x --train-batch 128 --opt-level O0
