"""""
This script saves 11 combinations consisting of:
- an OpinionGPT adapter (x 11)
- the Phi3 model as a base model

This is the first step towards bias combination with MergeKit.
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

#with open("../hugging_access_token.txt", "r") as file:
#    access_token = file.read().strip()

access_token = "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"

def save_models():
    model_id = "unsloth/Phi-3-mini-4k-instruct"
    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    device = "cuda:7"

    for adapter in adapters:
        print(f"Loading adapter: {adapter}")
        
        # Load a fresh copy of the base model every time
        base = AutoModelForCausalLM.from_pretrained(
            model_id, token=access_token
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=access_token
        )

        tmp_model = PeftModel.from_pretrained(
            base,
            lora_id.format(adapter=adapter),
            adapter_name=adapter,
            token=access_token
        ).to(device)

        merged_model = tmp_model.merge_and_unload()

        save_path = f"phi3_{adapter}"
        merged_model.save_pretrained(save_path)
        print(f"Saved model and tokenizer for adapter: {adapter}")

if __name__ == "__main__":
    save_models()
    print("Done saving model + adapter configurations.")
