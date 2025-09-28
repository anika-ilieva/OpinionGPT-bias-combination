"""
This script enables four-bias combination evaluation.
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

# Adjust paths as necessary
gender_models = [
    "phi3_women",
    "phi3_men",
]
political_models = [
    "phi3_liberal",
    "phi3_conservative",
]
age_models = [
    "phi3_teenagers",
    "phi3_old_people",
    "phi3_people_over_30",
]
geographic_models = [
    "phi3_american",
    "phi3_german",
    "phi3_latin_america",
    "phi3_middle_east",
]

# Paths to temporary stored configs and merged models
# Change paths as necessary
merge_repo = "mergekit"
output_dir = "mergekit/merged_combined_4"
config_dir = "mergekit/merge_configs_combined_4"
eval_repo = "ResponsibleNLP"
dataset_dir = os.path.join(eval_repo, "hbr_large")
results_dir = os.path.join(eval_repo, "result_combined_4")

# Error handling
failures = []
log_dir = os.path.join(config_dir, "logs_combined_4")
os.makedirs(log_dir, exist_ok=True)

# Create needed directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Search space
search_space = {
    "normalize": [True],
    "weight": [[0.5, 0.5, 0.5, 0.5]],
}

# Method and its parameters
method_params = {
    "linear": ["weight", "normalize"],
}

def ensure_weight_list(w, n_models=4):
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

        # Validate model-specific params for 4-model merge
        model_params = {}
        if "weight" in param_dict:
            model_params["weight"] = ensure_weight_list(param_dict["weight"], n_models=4)

        # Iterate over all quadruples: gender × political × age × geographic
        for gender_model, political_model, age_model, geo_model in itertools.product(
            gender_models, political_models, age_models, geographic_models
        ):
            # Per-model parameters (weights aligned by order)
            models_block = [
                {"model": gender_model,    "parameters": {"weight": model_params["weight"][0]}},
                {"model": political_model, "parameters": {"weight": model_params["weight"][1]}},
                {"model": age_model,       "parameters": {"weight": model_params["weight"][2]}},
                {"model": geo_model,       "parameters": {"weight": model_params["weight"][3]}},
            ]

            # YAML config
            config = {
                "models": models_block,
                "merge_method": method,
                "random_seed": 42,
            }

            # Base model: same behavior as before (not needed for linear)
            if method != "linear":
                config["base_model"] = base_model

            # Add top-level parameters
            if "normalize" in param_dict:
                config["normalize"] = param_dict["normalize"]

            # Unique tag
            gender_name = os.path.basename(gender_model)
            political_name = os.path.basename(political_model)
            age_name = os.path.basename(age_model)
            geo_name = os.path.basename(geo_model)

            w_tag = "w" + "-".join(str(w).replace(".", "") for w in model_params["weight"])
            w_human = "w" + "-".join(f"{w:.3f}" for w in model_params["weight"])
            tag = f"{method}_{gender_name}_{political_name}_{age_name}_{geo_name}_{w_tag}"

            # Paths
            config_path = os.path.join(config_dir, f"{tag}.yml")
            output_path = os.path.join(output_dir, tag)

            # Save YAML config
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f)

            # Logs
            merge_log = os.path.join(log_dir, f"{tag}_merge_combined_4.log")
            eval_log = os.path.join(log_dir, f"{tag}_eval_combined_4.log")

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
