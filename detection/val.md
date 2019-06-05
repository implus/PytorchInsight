SE
python tools/test.py local_configs/retinanet_r50_fpn_2x_pretrain_se_resnet50.py work_dirs/retinanet_r50_fpn_2x_pretrain_se_resnet50/epoch_24.pth --gpus 2 --out logs/val.retinanet_r50_fpn_2x_pretrain_se_resnet50.results.pkl --eval bbox > logs/val.retinanet_r50_fpn_2x_pretrain_se_resnet50
python tools/test.py local_configs/retinanet_r101_fpn_2x_pretrain_se_resnet101.py work_dirs/retinanet_r101_fpn_2x_pretrain_se_resnet101/epoch_24.pth --gpus 2 --out logs/val.retinanet_r101_fpn_2x_pretrain_se_resnet101.results.pkl --eval bbox > logs/val.retinanet_r101_fpn_2x_pretrain_se_resnet101


SGE
python tools/test.py local_configs/retinanet_r50_fpn_2x_pretrain_sge_resnet50.py work_dirs/retinanet_r50_fpn_2x_pretrain_sge_resnet50/epoch_24.pth --gpus 2 --out logs/val.retinanet_r50_fpn_2x_pretrain_sge_resnet50.results.pkl --eval bbox > logs/val.retinanet_r50_fpn_2x_pretrain_sge_resnet50
python tools/test.py local_configs/retinanet_r101_fpn_2x_pretrain_sge_resnet101.py work_dirs/retinanet_r101_fpn_2x_pretrain_sge_resnet101/epoch_24.pth --gpus 2 --out logs/val.retinanet_r101_fpn_2x_pretrain_sge_resnet101.results.pkl --eval bbox > logs/val.retinanet_r101_fpn_2x_pretrain_sge_resnet101

python tools/test.py local_configs/cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50.py work_dirs/cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50/epoch_20.pth --gpus 2 --out logs/val.cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50.results.pkl --eval bbox > logs/val.cascade_rcnn_r50_fpn_20e_pretrain_sge_resnet50
python tools/test.py local_configs/cascade_rcnn_r101_fpn_20e_pretrain_sge_resnet101.py work_dirs/cascade_rcnn_r101_fpn_20e_pretrain_sge_resnet101/epoch_20.pth --gpus 2 --out logs/val.cascade_rcnn_r101_fpn_20e_pretrain_sge_resnet101.results.pkl --eval bbox > logs/val.cascade_rcnn_r101_fpn_20e_pretrain_sge_resnet101

python tools/test.py local_configs/faster_rcnn_r101_fpn_2x_pretrain_sge_resnet101.py work_dirs/faster_rcnn_r101_fpn_2x_pretrain_sge_resnet101/epoch_24.pth --gpus 2 --out logs/val.faster_rcnn_r101_fpn_2x_pretrain_sge_resnet101.results.pkl --eval bbox > logs/val.faster_rcnn_r101_fpn_2x_pretrain_sge_resnet101
#mask_rcnn_r101_fpn_2x_pretrain_sge_resnet101
python tools/test.py local_configs/mask_rcnn_r101_fpn_2x_pretrain_sge_resnet101.py work_dirs/mask_rcnn_r101_fpn_2x_pretrain_sge_resnet101/epoch_24.pth --gpus 2 --out logs/val.mask_rcnn_r101_fpn_2x_pretrain_sge_resnet101.results.pkl --eval bbox > logs/val.mask_rcnn_r101_fpn_2x_pretrain_sge_resnet101


GC
python tools/test.py local_configs/retinanet_r50_fpn_2x_pretrain_gc_resnet50.py work_dirs/retinanet_r50_fpn_2x_pretrain_gc_resnet50/epoch_24.pth --gpus 2 --out logs/val.retinanet_r50_fpn_2x_pretrain_gc_resnet50.results.pkl --eval bbox > logs/val.retinanet_r50_fpn_2x_pretrain_gc_resnet50


CBAM
python tools/test.py local_configs/retinanet_r50_fpn_2x_pretrain_cbam_resnet50.py work_dirs/retinanet_r50_fpn_2x_pretrain_cbam_resnet50/epoch_24.pth --gpus 2 --out logs/val.retinanet_r50_fpn_2x_pretrain_cbam_resnet50.results.pkl --eval bbox > logs/val.retinanet_r50_fpn_2x_pretrain_cbam_resnet50
python tools/test.py local_configs/retinanet_r101_fpn_2x_pretrain_cbam_resnet101.py work_dirs/retinanet_r101_fpn_2x_pretrain_cbam_resnet101/epoch_24.pth --gpus 2 --out logs/val.retinanet_r101_fpn_2x_pretrain_cbam_resnet101.results.pkl --eval bbox > logs/val.retinanet_r101_fpn_2x_pretrain_cbam_resnet101


BAM
python tools/test.py local_configs/retinanet_r50_fpn_2x_pretrain_bam_resnet50.py work_dirs/retinanet_r50_fpn_2x_pretrain_bam_resnet50/epoch_24.pth --gpus 2 --out logs/val.retinanet_r50_fpn_2x_pretrain_bam_resnet50.results.pkl --eval bbox > logs/val.retinanet_r50_fpn_2x_pretrain_bam_resnet50


SK
python tools/test.py local_configs/retinanet_r50_fpn_2x_pretrain_sk_resnet50.py work_dirs/retinanet_r50_fpn_2x_pretrain_sk_resnet50/epoch_24.pth --gpus 2 --out logs/val.retinanet_r50_fpn_2x_pretrain_sk_resnet50.results.pkl --eval bbox > logs/val.retinanet_r50_fpn_2x_pretrain_sk_resnet50


old
python tools/test.py local_configs/cascade_rcnn_r50_fpn_20e.py work_dirs/cascade_rcnn_r50_fpn_20e/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth --gpus 2 --out logs/val.cascade_rcnn_r50_fpn_20e.results.pkl --eval bbox > logs/val.cascade_rcnn_r50_fpn_20e
python tools/test.py local_configs/cascade_rcnn_r101_fpn_20e.py work_dirs/cascade_rcnn_r101_fpn_20e/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth --gpus 2 --out logs/val.cascade_rcnn_r101_fpn_2x.results.pkl --eval bbox > logs/val.cascade_rcnn_r101_fpn_20e
python tools/test.py local_configs/mask_rcnn_r50_fpn_2x.py work_dirs/mask_rcnn_r50_fpn_2x/mask_rcnn_r50_fpn_2x_20181010-41d35c05.pth --gpus 2 --out logs/val.mask_rcnn_r50_fpn_2x.results.pkl --eval bbox > logs/val.mask_rcnn_r50_fpn_2x
python tools/test.py local_configs/mask_rcnn_r101_fpn_2x.py work_dirs/mask_rcnn_r101_fpn_2x/mask_rcnn_r101_fpn_2x_20181129-a254bdfc.pth --gpus 2 --out logs/val.mask_rcnn_r101_fpn_2x.results.pkl --eval bbox > logs/val.mask_rcnn_r101_fpn_2x
python tools/test.py local_configs/faster_rcnn_r101_fpn_2x.py work_dirs/faster_rcnn_r101_fpn_2x/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth --gpus 2 --out logs/val.faster_rcnn_r101_fpn_2x.results.pkl --eval bbox > logs/val.faster_rcnn_r101_fpn_2x
python tools/test.py local_configs/retinanet_r101_fpn_2x.py work_dirs/retinanet_r101_fpn_2x/epoch_24.pth --gpus 2 --out logs/val.retinanet_r101_fpn_2x.results.pkl --eval bbox > logs/val.retinanet_r101_fpn_2x
