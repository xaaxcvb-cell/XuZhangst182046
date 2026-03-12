#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=Dirichlet
#SBATCH --output=logs/job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch

echo "[INFO] Allocated node: $(hostname)"
cd ~/st182046/dirichlet/GMM

source ~/st182046/dirichlet/GMM/venv/bin/activate
# python3 -c "import jsonargparse; print('jsonargparse', jsonargparse.__version__)"

# Activate everything you need
module load cuda/10.1

CONFIG=~/st182046/dirichlet/GMM/configs/target_adaptation.yaml

# Run your python code
#echo "[INFO] Using Python: $(which python3)"
#python3 --version

#srun python3 ~/st182046/dirichlet/GMM/main.py

# Run your python code (for adaptation)
python main.py fit --config "$CONFIG"
#python main.py fit --config st182046/dirichlet/GMM/configs/target_adaptation.yaml