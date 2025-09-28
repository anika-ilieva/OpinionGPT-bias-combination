"""
This script enables three-bias combination evaluation.
"""

import argparse
import itertools
import os
import subprocess
import time
import traceback
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu-id",
    type=int,
    required=True,
    help="Which GPU to use (0–3)"
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Model IDs
base_model = "unsloth/Phi-3-mini-4k-instruct"

# Adjust models and paths as necessary
gender_models = [
    "phi3_women",
    "phi3_men",
]
age_models = [
    "phi3_teenagers",
    "phi3_old_people",
    "phi3_people_over_30",
]
geographic_models = [
    "phi3_middle_east",
    "phi3_german",
    "phi3_american",
    "phi3_latin_america",
]

# Adjust paths as necessary
merge_repo = "mergekit"
output_dir = "mergekit/merged_gender_age_geographic"
config_dir = "mergekit/merge_gender_age_geographic"
eval_repo = "ResponsibleNLP"
dataset_dir = os.path.join(eval_repo, "hbr_large")
results_dir = os.path.join(eval_repo, "result_gender_age_geographic")

# Error handling
failures = []
log_dir = os.path.join(config_dir, "logs_gender_age_geographic")
os.makedirs(log_dir, exist_ok=True)

#  Create needed directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Search space
search_space = {
    "normalize": [True],
    "weight": [[0.5, 0.5, 0.5]],
}

# Method parameters
method_params = {
    "linear": ["weight", "normalize"],
}

def ensure_weight_list(w, n_models=3):
    """
    Normalize weight parameter:
    - If float/int -> repeat to length n_models
    - If list -> validate length == n_models
    """
    if isinstance(w, (int, float)):
        return [float(w)] * n_models
    if isinstance(w, list):
        if len(w) != n_models:
            raise ValueError(f"'weight' must have length {n_models} for {n_models}-model merge, got {len(w)}: {w}")
        return [float(x) for x in w]
    raise TypeError(f"Unsupported 'weight' type: {type(w)}")

# Main loop
for method, param_keys in method_params.items():
    param_values = [search_space[p] for p in param_keys]

    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_keys, values))

        # Validate model-specific params for 3-model merge
        model_params = {}
        if "weight" in param_dict:
            model_params["weight"] = ensure_weight_list(param_dict["weight"], n_models=3)

        # Iterate over all triplets: gender × geographic × age
        for gender_model, geographic_model, age_model in itertools.product(gender_models, geographic_models, age_models):
            # Construct per-model parameter blocks
            # Each model gets the *same* per-model params.
            # MergeKit linear will align these by order.
            models_block = [
                {"model": gender_model, "parameters": {"weight": model_params["weight"][0]}},
                {"model": geographic_model, "parameters": {"weight": model_params["weight"][1]}},
                {"model": age_model, "parameters": {"weight": model_params["weight"][2]}},
            ]

            # YAML config
            config = {
                "models": models_block,
                "merge_method": method,
                "random_seed": 42,
            }

            # Base model: keep same behavior as before
            if method != "linear":
                config["base_model"] = base_model

            # Add top-level parameters
            if "normalize" in param_dict:
                config["normalize"] = param_dict["normalize"]

            # Unique tag
            gender_name = os.path.basename(gender_model)
            geographic_name = os.path.basename(geographic_model)
            age_name = os.path.basename(age_model)

            # Flatten weights without dots for filenames, but also include readable weights for clarity
            w_tag = "w" + "-".join(str(w).replace(".", "") for w in model_params["weight"])
            w_human = "w" + "-".join(f"{w:.3f}" for w in model_params["weight"])
            tag = f"{method}_{gender_name}_{geographic_name}_{age_name}_{w_tag}"
            config_path = os.path.join(config_dir, f"{tag}.yml")
            output_path = os.path.join(output_dir, tag)

            # Save YAML config
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

            # Logs
            merge_log = os.path.join(log_dir, f"{tag}_merge_gender_age_geographic.log")
            eval_log = os.path.join(log_dir, f"{tag}_eval_gender_age_geographic.log")

            start_time = time.time()

            # Merge step
            try:
                with open(merge_log, "w") as log_f:
                    subprocess.run(
                        [
                            "mergekit-yaml",
                            config_path,
                            output_path,
                            "--lazy-unpickle",
                            "--allow-crimes",
                            "--random-seed", "42",
                            "--safe-serialization",
                            "--write-model-card",
                            "--trust-remote-code",
                            "--cuda",
                        ],
                        cwd=merge_repo,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            except subprocess.CalledProcessError:
                failures.append((tag, "merge"))
                print(f"[ERROR] Merge failed for: {tag} ({w_human})")
                traceback.print_exc()
                continue

            # Evaluation step
            try:
                with open(eval_log, "w") as log_f:
                    subprocess.run(
                        [
                            "python", "-m", "robbie.eval",
                            "--dataset", "holisticbiasr",
                            "--model-id", output_path,
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
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            except subprocess.CalledProcessError:
                failures.append((tag, "eval"))
                print(f"[ERROR] Evaluation failed for: {tag} ({w_human})")
                traceback.print_exc()
            finally:
                # Cleanup merged model to save space
                try:
                    if os.path.exists(output_path):
                        subprocess.run(["rm", "-rf", output_path])
                except Exception:
                    print(f"[WARNING] Could not delete {output_path}")

            # Timing
            elapsed = time.time() - start_time
            mins, secs = divmod(elapsed, 60)
            print(f"[INFO] Finished {tag} ({w_human}) in {int(mins)}m {int(secs)}s.")

# Summary
if failures:
    print("\n========== SUMMARY OF FAILURES ==========")
    for tag, stage in failures:
        print(f"{tag} failed during {stage}")
    print("=========================================\n")
else:
    print("\nAll merges and evaluations completed successfully!\n")

