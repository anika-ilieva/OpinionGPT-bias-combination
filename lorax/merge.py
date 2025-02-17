from lorax import Client, MergedAdapters
import json

with open("../hugging_access_token.txt", "r") as file:
    access_token = file.read().strip()

# LoRAX endpoint (change if necessary)
endpoint_url = "http://127.0.0.1:8080"
client = Client(endpoint_url)

adapters = [
    "HU-Berlin-ML-Internal/opiniongpt-phi3-american",
    "HU-Berlin-ML-Internal/opiniongpt-phi3-women",
]
bias_to_subreddit = {
    "american": "AskAnAmerican",
    "women": "AskWomen",
}

# Define weights and merge strategy
merged_adapters = MergedAdapters(
    ids=adapters,
    weights=[0.5, 0.5],
    merge_strategy="ties", 
    density=0.2,
    majority_sign_method="total",
)

# Create the instruction template
instruction_template = (
    "### r/{subreddit1} r/{subreddit2} Question:\n\n"
    "{instruction}\n\n"
    "### r/{subreddit1} r/{subreddit2} Answer:\n\n"
)

# Interactive inference
def inference():
    while True:
        try:
            subreddit1 = bias_to_subreddit["american"]
            subreddit2 = bias_to_subreddit["women"]
            instruction = input("Enter instruction: ")

            prompt = instruction_template.format(
                subreddit1=subreddit1,
                subreddit2=subreddit2,
                instruction=instruction
            )

            # Send request to LoRAX with merged adapters
            response = client.generate(
                prompt,
                merged_adapters=merged_adapters,
                api_token=access_token,
            )
            print(response.generated_text)


        except KeyboardInterrupt:
            print("Exiting...")
            exit()

if __name__ == "__main__":
    print("Done loading adapters. Starting inference...")
    inference()
