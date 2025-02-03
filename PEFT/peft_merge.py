"""""
This script enables merging of OpinionGPT adapers via PEFT.
The two merging techniques available are TIES and DATRE-TIES.
"""""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapters = [
    "american",
    "women",
]

bias_to_subreddit = {
    "american": "AskAnAmerican",
    "women": "AskWomen",
}

with open("../hugging_access_token.txt", "r") as file:
    access_token = file.read().strip()

def merge_models():
    model_id = "unsloth/Phi-3-mini-4k-instruct"
    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    model = AutoModelForCausalLM.\
        from_pretrained(model_id, token=access_token)
    tokenizer = AutoTokenizer.\
        from_pretrained(model_id, token=access_token)
    adapters = ["american", "women"]
    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model,
        lora_id.format(adapter=default_adapter),
        adapter_name=default_adapter,
        token=access_token
    )

    for adapter in adapters[1:]:
        print("Loading adapter: {}".format(adapter))
        model.load_adapter(
            lora_id.format(adapter=adapter),
            adapter,
            token=access_token
        )

    # Apply merging method. 
    # Set combination_type to "ties" for TIES merging and "dare_ties" 
    # for DATRE-TIES merging.
    weights = [2.0, 1.0]
    adapter_name = "merged"
    density = 0.2
    model.add_weighted_adapter(
        adapters,
        weights,
        adapter_name,
        combination_type="dare_ties",
        density=density
    )

    # Set the newly merged adapter as the active adapter with the 
    # set_adapter() method.
    model.set_adapter("merged")
    model.save_pretrained("merged")
    
    return model, tokenizer

def inference(model, tokenizer):
    instruction_template = (
        "### r/{subreddit1} r/{subreddit2} Question:\n\n"
        "{instruction}\n\n"
        "### r/{subreddit1} r/{subreddit2} Answer:\n\n"
    )

    while True:
        try:
            subreddit1 = bias_to_subreddit["american"]
            subreddit2 = bias_to_subreddit["women"]
            instruction = input("Enter instruction: ")
            prompt = instruction_template.format(
                subreddit1=subreddit1,
                subreddit2=subreddit2,
                instruction=instruction)

            inputs = tokenizer(prompt, return_tensors="pt")\
                .input_ids.to(model.device)

            generation_kwargs = dict(
                input_ids=inputs,
                max_new_tokens=256,
                min_new_tokens=10,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=0.8
            )

            output = model.generate(**generation_kwargs)
            print(tokenizer.decode(output[0], skip_special_tokens=True))

        except KeyboardInterrupt:
            print("Exiting...")
            exit()

if __name__ == "__main__":
    model, tokenizer = merge_models()
    print("Done loading model.")
    inference(model, tokenizer)