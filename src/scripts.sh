# MovieLens
## MF
CUDA_VISIBLE_DEVICES=0 python main.py -d MovieLens

## MF + LCD-UC
CUDA_VISIBLE_DEVICES=0 python main.py -d MovieLens --box -attn --mask

# KwaiRec
## MF
CUDA_VISIBLE_DEVICES=1 python main.py -d KuaiRec -lr 0.01

## MF + LCD-UC
CUDA_VISIBLE_DEVICES=1 python main.py -d KuaiRec --box -attn --mask -lr 0.01