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

with open("../hugging_access_token.txt", "r") as file:
    access_token = file.read().strip()

def save_models():
    model_id = "unsloth/Phi-3-mini-4k-instruct"
    lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
    device = "cuda:2"

    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=access_token
    )

    # Iterate over all adapters - load, set, and save them
    for adapter in adapters:
        print(f"Loading adapter: {adapter}")
        tmp_model = PeftModel.from_pretrained(
            base_model,
            lora_id.format(adapter=adapter),
            adapter_name=adapter,
            token=access_token
        ).to(device)

        # Set current adapter 
        #tmp_model.set_adapter(adapter)
        merged_model = tmp_model.merge_and_unload()

        # Save model + current adapter combination
        save_path = f"phi3_{adapter}"
        merged_model.save_pretrained(save_path)
        #tokenizer.save_pretrained(save_path)

        print(f"Saved model and tokenizer for adapter: {adapter}")

if __name__ == "__main__":
    save_models()
    print("Done saving model + adapter configurations.")
