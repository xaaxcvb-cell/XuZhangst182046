python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_JOB_GPUS =", os.environ.get("SLURM_JOB_GPUS"))
print("SLURM_STEP_GPUS =", os.environ.get("SLURM_STEP_GPUS"))
print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)

try:
    print("is_available =", torch.cuda.is_available())
    print("device_count =", torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print("current_device =", torch.cuda.current_device())
        print("device_name =", torch.cuda.get_device_name(0))
except Exception as e:
    print("CUDA test failed:", repr(e))
PY