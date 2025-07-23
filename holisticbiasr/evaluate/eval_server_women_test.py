"""""
This script enables the evaluation of different bias combinations
on the HolisticBiasR large dataset. 

How to run:
1. Naviagate to /vol/tmp/ilievaan/ResponsibleNLP
2. Create a conda env by running the following at the root of the repo: 
conda create -n robbie -y python=3.10.6
conda activate robbie
pip install .
pip install -r robbie/requirements.txt
3. Naviagate to /vol/tmp/ilievaan/OpinionGPT
4. Edit the access_token variable in this script to your Hugging Face token
and upload the script to the server in the appropriate directory 
(/vol/tmp/ilievaan/OpinionGPT).
5. Run the script with the command:
nohup python3 eval_server_women_test.py --gpu-id X > mylog.txt 2>&1 &
"""""

import argparse
import os
import subprocess
import time
import traceback

# Hugging Face token for model access
access_token = "xxx" # Replace with your actual token

# Parse which GPU to use
parser = argparse.ArgumentParser()
parser.add_argument("--gpu-id", type=int, required=True, help="Which GPU to use (0–3)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Evaluation config
all_models = ["HU-Berlin-ML-Internal/phi3_women"]
eval_repo = "/vol/tmp/ilievaan-pub/ResponsibleNLP"
dataset_dir = os.path.join(eval_repo, "v1medium")
results_dir = os.path.join(eval_repo, "result_phi3_women_test")
log_dir = os.path.join(results_dir, "logs_phi3_women_test")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Error tracking
failures = []

# Run evaluation
for model_id in all_models:
    tag = os.path.basename(model_id)
    eval_log = os.path.join(log_dir, f"{tag}_eval_women_test.log")

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

if failures:
    print("\n========== SUMMARY OF FAILURES ==========")
    for tag in failures:
        print(f"{tag} failed during evaluation")
    print("=========================================\n")
else:
    print("\nAll evaluations completed successfully!\n")
