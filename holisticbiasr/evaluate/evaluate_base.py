"""""
This script enables the evaluation of OpinionGPT base biases and
the Phi-3 model on the HolisticBiasR large dataset.
"""""

import argparse
import os
import subprocess
import time
import traceback

# Hugging Face token for model access
with open("hugging_access_token.txt", "r") as file:
    access_token = file.read().strip()

# Parse which GPU to use
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True, help="Which GPU to use (0–3)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Model lists
# Change paths as necessary
all_models = [
    "unsloth/Phi-3-mini-4k-instruct",
    # Edit paths to biased models (LoRA adapters already applied to base model)
    "phi3_teenagers",
    "phi3_old_people",
    "phi3_people_over_30",
    "phi3_middle_east",
    "phi3_german",
    "phi3_american",
    "phi3_latin_america",
    "phi3_liberal",
    "phi3_conservative",
    "phi3_men",
    "phi3_women"
]

# Evaluation config
# Change paths as necessary
eval_repo = "ResponsibleNLP"
dataset_dir = os.path.join(eval_repo, "hbr_large")
results_dir = os.path.join(eval_repo, "result_evaluate_base")
log_dir = os.path.join(results_dir, "logs_evaluate_base")

# Ensure necessary dirs
os.makedirs(results_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Error tracking
failures = []

# Run evaluation
for model_id in all_models:
    tag = os.path.basename(model_id)
    eval_log = os.path.join(log_dir, f"{tag}_evaluate_base.log")

    start_time = time.time()
    try:
        env = os.environ.copy()
        env["HUGGING_FACE_HUB_TOKEN"] = access_token
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
                    "--max-length", "64",
                    "--batch-size", "32",
                ],
                cwd=eval_repo,
                stdout=log_f, stderr=subprocess.STDOUT,
                check=True,
                env=env
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
