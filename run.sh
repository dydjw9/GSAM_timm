#GSAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py /home/djw/data/imagenet --model vit_small_patch32_224 -b 512 --lr 3e-3 --sched cosine --epochs 300  --model-ema --opt adamw -j 8 --warmup-lr 1e-6 --weight-decay 0.3 --drop 0.0 --drop-path .1 --warmup-epochs 5 --img-size 224   --smoothing 0.1 --native-amp
#LookSAM
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lsam.py /home/djw/data/imagenet --model resnet152 --sched cosine  -b 51 --lr 3e-3 --sched cosine --epochs 300  --model-ema --weight-decay 0.3 --img-size 224   --smoothing 0.1 --native-amp
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lsam.py /home/djw/data/imagenet --model resnet152 --sched cosine  -b 51 --lr 3e-3 --sched cosine --epochs 300  --model-ema --weight-decay 0.3 --img-size 224   --smoothing 0.1 --native-amp
