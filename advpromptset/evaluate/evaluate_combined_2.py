"""""
This script enables the evaluation of combinations of 
two OpinionGPT biases on the full AdvPromptSet dataset.
"""""

import argparse
import itertools
import os
import subprocess
import time
import traceback
import yaml

# Parse which GPU to use
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True, help="Which GPU to use (0–3)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Model IDs
base_model = "unsloth/Phi-3-mini-4k-instruct"
# Adjust paths to biased models (LoRA adapters already applied to base model)
gender_models = [
    "phi3_men",
    "phi3_women"
]
political_models = [
    "phi3_conservative",
    "phi3_liberal"
]

# Paths to temporary stored configs and merged models
# Adjust paths as necessary
merge_repo = "mergekit"
output_dir = "mergekit/merged_gender_political"
config_dir = "mergekit/merge_gender_political"

# Path to ResponsibleNLP repo and dataset
# Adjust paths as necessary
eval_repo = "ResponsibleNLP"
dataset_dir = "ResponsibleNLP/AdvPromptSet"
results_dir = "ResponsibleNLP/result_gender_political"

# Error handling
failures = []
log_dir = os.path.join(config_dir, "logs_gender_political")
os.makedirs(log_dir, exist_ok=True)

# Create needed directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Search space for optimal parameter values
search_space = {
    "normalize": [True],
    "weight": [[0.5,0.5]],
}

# Merging method
method_params = {
    "linear": ["weight", "normalize"],
}

# Iterate over all combinations of
# two biases + merging technique + parameter value
for method, param_keys in method_params.items():
    param_values = [search_space[p] for p in param_keys]

    # Loop over all parameter combinations FIRST
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_keys, values))

        # Extract only model-specific parameters
        model_params = {
            k: param_dict[k]
            for k in ["weight"]
            if k in param_dict
        }

        # Now loop over all model combinations
        for political_model in political_models:
            for gender_model in gender_models:
                models_block = [
                    {"model": political_model, "parameters": model_params},
                    {"model": gender_model, "parameters": model_params}
                ]
                
                # Construct configuration for YAML file
                config = {
                    "models": models_block,
                    "merge_method": method,
                    "random_seed": 42
                }

                # Add base model if required
                if method != "linear":
                    config["base_model"] = base_model

                # Add top-level parameters
                for key in ["normalize"]:
                    if key in param_dict:
                        config[key] = param_dict[key]

                # Generate unique tag
                political_name = os.path.basename(political_model)
                gender_name = os.path.basename(gender_model)
                tag = f"{method}_{political_name}_{gender_name}_" + "_".join(
                    f"{k}{str(v).replace('.', '')}" for k, v in param_dict.items())

                # Paths
                config_path = os.path.join(config_dir, f"{tag}.yml")
                output_path = os.path.join(output_dir, tag)

                # Save YAML config
                with open(config_path, "w") as f:
                    yaml.safe_dump(config, f)

                # Log file paths
                merge_log = os.path.join(log_dir, f"{tag}_merge_gender_political.log")
                eval_log = os.path.join(log_dir, f"{tag}_eval_gender_political.log")

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
                                "--cuda"
                            ],
                            cwd=merge_repo,
                            stdout=log_f, stderr=subprocess.STDOUT,
                            check=True
                        )
                except subprocess.CalledProcessError:
                    failures.append((tag, "merge"))
                    print(f"[ERROR] Merge failed for: {tag}")
                    traceback.print_exc()
                    continue  # skip to next configuration

                # Evaluation step
                try:
                    with open(eval_log, "w") as log_f:
                        subprocess.run(
                            [
                                "python", "-m", "robbie.eval",
                                "--dataset", "advpromptset",
                                "--model-id", output_path,
                                "--metric", "toxigen",
                                "--dataset-dir", dataset_dir,
                                "--predictor", "hf_causal",
                                "--device", "cuda",
                                "--result-dir", results_dir,
                                "--seed", "42",
                                "--max-length", "64",
                                "--batch-size", "32",
                                "--top-k", "50"
                            ],
                            cwd=eval_repo,
                            stdout=log_f, stderr=subprocess.STDOUT,
                            check=True,
                        )
                except subprocess.CalledProcessError:
                    failures.append((tag, "eval"))
                    print(f"[ERROR] Evaluation failed for: {tag}")
                    traceback.print_exc()
                    continue  # skip to next configuration

                # Cleanup
                try:
                    if os.path.exists(output_path):
                        subprocess.run(["rm", "-rf", output_path])
                except Exception:
                    print(f"[WARNING] Could not delete {output_path}")

                # Print last runtime
                end_time = time.time()
                elapsed = end_time - start_time
                mins, secs = divmod(elapsed, 60)
                print(f"[INFO] Finished {tag} in {int(mins)}m {int(secs)}s.")

# Print failure summary
if failures:
    print("\n========== SUMMARY OF FAILURES ==========")
    for tag, stage in failures:
        print(f"{tag} failed during {stage}")
    print("=========================================\n")
else:
    print("\nAll merges and evaluations completed successfully!\n")