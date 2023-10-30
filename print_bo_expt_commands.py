from pathlib import Path

MAX_THREADS = 6
target = "F2"
out_path = Path("results") / "bo" / target
out_path.mkdir(parents=True, exist_ok=True)
for N in [
    500,
    1000,
    2000,
    4000,
    8000,
    16_000,
]:
    for trial in range(5):
        trial += 1  # start from 1
        command = (
            f"OMP_NUM_THREADS={MAX_THREADS} OPENBLAS_NUM_THREADS={MAX_THREADS} MKL_NUM_THREADS={MAX_THREADS} "
            f"VECLIB_MAXIMUM_THREADS={MAX_THREADS} NUMEXPR_NUM_THREADS={MAX_THREADS} "
            "CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:$PYTHONPATH python "
            "experiment_scripts/bo_with_thompson_sampling.py "
            f"--seed={trial} --target={target} --batch_size=100 --dataset_size={N} --num_jobs={MAX_THREADS} "
            "--num_random_features=5000 "
            f"--output_json={out_path}/N{N:06d}-{trial}.json "
        )
        print(command)
