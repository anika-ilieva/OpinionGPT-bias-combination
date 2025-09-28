"""""
This script enables inference with the Phi3 model combined 
with multiple OpinionGPT adapters.
"""""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapters = [
    "american",
    "conservative",
    "german",
    "latin_america",
    "liberal",
    "men",
    "middle_east",
    "old_people",
    "people_over_30",
    "teenagers", 
    "women",
]

bias_to_subreddit = {
    "liberal": "AskALiberal",
    "conservative": "AskConservatives",
    "german": "AskAGerman",
    "american": "AskAnAmerican",
    "latin_american": "AskLatinAmerica",
    "middle_east": "AskMiddleEast",
    "men": "AskMen",
    "women": "AskWomen",
    "people_over_30": "AskPeopleOver30",
    "old_people": "AskOldPeople",
    "teenager": "AskTeenagers",
}

with open("hugging_access_token.txt", "r") as file:
    access_token = file.read().strip()

def load_model():
    model_id = "unsloth/Phi-3-mini-4k-instruct"
    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    device = "cuda:2"
    model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

    default_adapter = adapters[0]

    model = PeftModel.from_pretrained(
        model,
        lora_id.format(adapter=default_adapter),
        adapter_name=default_adapter,
        token=access_token
    ).to(device)

    for adapter in adapters[1:]:
        print("Loading adapter: {}".format(adapter))
        model.load_adapter(
            lora_id.format(adapter=adapter),
            adapter,
            token=access_token
        )

    target_adapter = "middle_east"
    
    if model.active_adapter != "middle_east":
        model.set_adapter(target_adapter)

    return model, tokenizer

def inference(model, tokenizer):
    instruction_template = (
        "### r/{subreddit} Question:\n\n"
        "{instruction}\n\n"
        "### r/{subreddit} Answer:\n\n"
    )

    while True:
        try:
            subreddit = bias_to_subreddit[model.active_adapter]
            instruction = input("Enter instruction: ")
            prompt = instruction_template.format(
                subreddit=subreddit,
                instruction=instruction
            )

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
    model, tokenizer = load_model()
    print("Done loading model.")
    inference(model, tokenizer)
