"""""
This script evaluates four OpinionGPT biases 
on the sampled HolisticBiasR dataset.
"""""

import argparse
import os
import subprocess
import time
import traceback

# Parse which GPU to use
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True, help="Which GPU to use (0–3)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Model lists
# Adjust paths as needed
all_models = [
    "phi3_liberal",
    "phi3_conservative",
    "phi3_women",
    "phi3_men"
]

# Evaluation config
# Adjust paths as needed
eval_repo = "ResponsibleNLP"
dataset_dir = os.path.join(eval_repo, "hbr_sampled")
results_dir = os.path.join(eval_repo, "result_grid_search_base")
log_dir = os.path.join(results_dir, "logs_grid_search_base")

# Ensure necessary dirs
os.makedirs(results_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Error tracking
failures = []

# Run evaluation
for model_id in all_models:
    tag = os.path.basename(model_id)
    eval_log = os.path.join(log_dir, f"{tag}_grid_search_base.log")

    start_time = time.time()
    try:
        with open(eval_log, "w") as log_f:
            subprocess.run(
                [
                    "python", "-m", "robbie.eval",
                    "--dataset", "holisticbiasr",
                    "--model-id", model_id,
                    "--metric", "regard",
                    "--dataset-dir", dataset_dir,
                    "--predictor", "hf_causal",
                    "--device", "cuda",
                    "--result-dir", results_dir,
                    "--seed", "42",
                ],
                cwd=eval_repo,
                stdout=log_f, stderr=subprocess.STDOUT,
                check=True
            )
    except subprocess.CalledProcessError:
        failures.append(tag)
        print(f"[ERROR] Evaluation failed for: {tag}")
        traceback.print_exc()
        continue

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"[INFO] Finished {tag} in {int(mins)}m {int(secs)}s.")

# Print failure summary
if failures:
    print("\n========== SUMMARY OF FAILURES ==========")
    for tag in failures:
        print(f"{tag} failed during evaluation")
    print("=========================================\n")
else:
    print("\nAll evaluations completed successfully!\n")
