"""""
This script enables merging of OpinionGPT adapers via PEFT.
The two merging techniques available are TIES and DATRE-TIES.
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

with open("../hugging_access_token.txt", "r") as file:
    access_token = file.read().strip()

model_id = "unsloth/Phi-3-mini-4k-instruct"
lora_id = "HU-Berlin-ML-Internal/opiniongpt-phi3-{adapter}"
model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

default_adapter = adapters[0]

model = PeftModel.from_pretrained(model, lora_id.format(adapter=default_adapter), \
                                    adapter_name=default_adapter, token=access_token)

for adapter in adapters[1:]:
    print("Loading adapter: {}".format(adapter))
    model.load_adapter(lora_id.format(adapter=adapter), adapter, token=access_token)

# Apply merging method. 
# Set combination_type to "ties" for TIES merging and "dare_ties" for DATRE-TIES merging.
adapters = ["american", "women"]
weights = [2.0, 1.0]
adapter_name = "merged"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)

# Set the newly merged adapter as the active adapter with the set_adapter() method.
model.set_adapter("merged")
print("Done merging.")