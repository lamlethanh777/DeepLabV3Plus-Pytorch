#!/bin/bash
#SBATCH --job-name=e4-voc
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=11:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch/experiment/e3/slurm_logs/%j_%x.out  # output
#SBATCH --error=/media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch/experiment/e3/slurm_logs/%j_%x.err   # error

# START
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "My SLURM_JOB_ID is ${SLURM_JOB_ID}"
echo "EXECUTING ON MACHINE:" $(hostname)
echo "START TIME: " $(date)
START_TIME=$(date +%s)
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

################################# MAIN PART #################################
# ENV
source /media02/tphung/third-parties/miniconda3/bin/activate vlm-old

# VARIABLE
PROJECT_ROOT="/media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch"
EXP_NAME="e4"
DATASET="voc"
CONFIG_FILE="experiment/${EXP_NAME}/deeplabv3plus_mobilenet_v3_large_attention_${DATASET}_improved.yaml"
MODEL_CHECKPOINT="${PROJECT_ROOT}/experiment/${EXP_NAME}/checkpoints/best_deeplabv3plus_mobilenet_v3_large_attention_${DATASET}_os16.pth"

cd $PROJECT_ROOT
pwd

# Any additional command-line arguments will override the config file
python main.py \
    --config $CONFIG_FILE \
    --exp_name $EXP_NAME \
    --batch_size 16 \
    --val_batch_size 1 \
    # --ckpt $MODEL_CHECKPOINT \
    # --test_only
    # --save_val_results \
    # --continue_training \

################################# MAIN PART #################################

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
