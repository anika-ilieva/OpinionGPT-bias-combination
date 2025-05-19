"""""
This script enables the tuning of different merging techniques
through grid search over selected parameter values.

Evaluation design
Given two demographic groups of biases - gender (women, men)
and political (liberal, conservative).
For each combination of biases, technique and parameter values:
1. Use MergeKit to merge the adapters
2. Temporary save the merged adapter
3. Evaluate the merged adapter using the small version of HolisticBiasR 
4. Save the evaluation results, delete the temporary adapter, proceed
"""""

import itertools
import os
import subprocess
import time
import traceback
import yaml

access_token = "XXX"

# Model IDs
base_model = "unsloth/Phi-3-mini-4k-instruct"
political_models = [
    "HU-Berlin-ML-Internal/phi3_liberal",
    "HU-Berlin-ML-Internal/phi3_conservative"
]
gender_models = [
    "HU-Berlin-ML-Internal/phi3_women",
    "HU-Berlin-ML-Internal/phi3_men"
]

# Paths to temporary stored configs and merged models
merge_repo = "/vol/tmp/ilievaan/mergekit"
output_dir = "/vol/tmp/ilievaan/mergekit/merged_models"
config_dir = "/vol/tmp/ilievaan/mergekit/merge_configs"

# Path to ResponsibleNLP repo and dataset
eval_repo = "/vol/tmp/ilievaan/ResponsibleNLP"
dataset_dir = os.path.join(eval_repo, "v1small")
results_dir = os.path.join(eval_repo, "result_grid")

# Error handling
failures = []
log_dir = os.path.join(config_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Create needed directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Search space for optimal parameter values
search_space = {
    "normalize": [True, False],
    "weight": [[0.3,0.3],[0.5,0.5],[0.7,0.7]], # [0.7,0.3], [0.3,0.7]
    "lambda": [0.05, 1, 1.5],
    "density": [0.5, 0.7, 0.9],
    "gamma": [0.01, 0.05, 0.1],
    "epsilon":  [0.01, 0.05, 0.1],
    "filter_wise": [False], # True is not recommended
    "select_top_k": [0.1, 0.5]
}
# Parameters of irrelevant methods are not included in the search space
# "t_slerp": [0.1, 0.3, 0.5, 0.7, 0.9],
# "t_nearswap": [0.01, 0.1, 0.3, 0.5],
# "nuslerp_flatten": [True, False],
# "nuslerp_row_wise": [True, False],

# Merging methods that support multi-model merging and use base model
method_params = {
    "linear": ["weight", "normalize"],
    "task_arithmetic": ["weight", "normalize", "lambda"],
    "ties": ["weight", "normalize", "lambda", "density"],
    "dare_linear": ["weight", "normalize"],
    "dare_ties": ["weight", "normalize", "lambda", "density"],
    "breadcrumbs": ["weight", "normalize", "lambda", "density", "gamma"],
    "breadcrumbs_ties": ["weight", "normalize", "lambda", "density", "gamma"],
    "della": ["weight", "normalize", "lambda", "density", "epsilon"],
    "della_linear": ["weight", "normalize", "lambda", "density", "epsilon"],
    "sce": ["weight", "normalize", "lambda", "density", "select_top_k"],
    "model_stock": ["filter_wise"]
}
# Other methods: slerp, nearswap, nuslerp, passthough

# Iterate over all combinations of
# two biases + merging technique + parameter value
# Outer loop: methods (one at a time)
for method, param_keys in method_params.items():
    param_values = [search_space[p] for p in param_keys]

    # Loop over all parameter combinations FIRST
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_keys, values))

        # Extract only model-specific parameters
        model_params = {
            k: param_dict[k]
            for k in ["weight", "density", "gamma", "epsilon"]
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
                for key in ["normalize", "lambda",
                            "filter_wise", "select_top_k"]:
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
                merge_log = os.path.join(log_dir, f"{tag}_merge.log")
                eval_log = os.path.join(log_dir, f"{tag}_eval.log")

                start_time = time.time()
                # Merge step
                try:
                    env = os.environ.copy()
                    env["HUGGING_FACE_HUB_TOKEN"] = access_token
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
                            check=True,
                            env=env
                        )
                except subprocess.CalledProcessError:
                    failures.append((tag, "merge"))
                    print(f"[ERROR] Merge failed for: {tag}")
                    traceback.print_exc()
                    continue  # skip to next configuration

                # Evaluation step
                try:
                    env = os.environ.copy()
                    env["HUGGING_FACE_HUB_TOKEN"] = access_token
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
                                "--num-samples", "1000",
                                "--seed", "42"
                            ],
                            cwd=eval_repo,
                            stdout=log_f, stderr=subprocess.STDOUT,
                            check=True,
                            env=env
                        )
                except subprocess.CalledProcessError:
                    failures.append((tag, "eval"))
                    print(f"[ERROR] Evaluation failed for: {tag}")
                    traceback.print_exc()
                    continue  # skip to next configuration

                # Cleanup
                try:
                    env = os.environ.copy()
                    env["HUGGING_FACE_HUB_TOKEN"] = access_token
                    if os.path.exists(output_path):
                        subprocess.run(["rm", "-rf", output_path], env=env)
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