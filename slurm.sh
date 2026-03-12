#!/bin/bash -l
#SBATCH --job-name=st182046_Dirichlet
#SBATCH --output=logs/job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --qos=batch

set -euxo pipefail

echo "SCRIPT_PATH=$0"
pwd
date

echo "HEAD_OF_SCRIPT:"
head -n 40 "$0" || true

module load cuda/10.1

PY="/no_backups/s1501/.pyenv/versions/3.10.13/envs/venv/bin/python"
echo "FORCED_PY=$PY"
ls -l "$PY"
"$PY" -V
"$PY" -c "import sys; print('exe=', sys.executable)"

"$PY" -m pip -V
"$PY" -m pip install -U "jsonargparse[signatures]>=4.27.7"

exec "$PY" main.py