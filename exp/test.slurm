#!/bin/bash
#SBATCH --job-name=v3-101-cityscapes
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch/exp/slurm_logs/%j_%x.out  # output
#SBATCH --error=/media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch//exp/slurm_logs/%j_%x.err   # error

# START
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "My SLURM_JOB_ID is ${SLURM_JOB_ID}"
echo "EXECUTING ON MACHINE:" $(hostname)
echo "START TIME: " $(date)
START_TIME=$(date +%s)
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

# load miniconda
source /media02/tphung/third-parties/miniconda3/bin/activate vlm-old

# run code
cd /media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch
pwd

python main.py \
    --model deeplabv3plus_mobilenet \
    --gpu_id 0 \
    --dataset cityscapes \
    --data_root datasets/data/cityscapes \
    --crop_val \
    --lr 0.01 \
    --crop_size 513 \
    --batch_size 16 \
    --output_stride 16 \
    --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth \
    --test_only \
    --save_val_results \
    --ckpt checkpoints/

# END
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "My SLURM_JOB_ID is ${SLURM_JOB_ID}"
echo "END TIME: " $(date)
END_TIME=$(date +%s)

duration=$(($END_TIME - $START_TIME))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))
echo "DURATION: ${hours}h ${minutes}m ${seconds}s"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"