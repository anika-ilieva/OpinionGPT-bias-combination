# AdvPromptSet Evaluation

> Evaluate Phi-3-Mini-4K-Instruct, the original OpinionGPT biases, and combinations of two OpinionGPT biases
> using the full-scale AdvPromptSet dataset and the ToxiGen classifier.

## Base model and original OpinionGPT biases evaluation

> These experiments do not include a model merging step.

- Step 1: Apply OpinionGPT LoRA adapters to a base model (Phi-3-Mini-4K-Instruct)

Navigate to utils and execute [save_models.py](https://github.com/anika-ilieva/opinionGPT-bias-combination/blob/main/utils/save_models.py)

~~~~
python3 save_models.py
~~~~

- Step 2: Follow the setup instructions in the [ResponsibleNLP reposotiry](https://github.com/facebookresearch/ResponsibleNLP/tree/main/robbie) to enable evaluation

Note 1: Go to [advpromptset.py](https://github.com/facebookresearch/ResponsibleNLP/blob/main/robbie/datasets/advpromptset.py)
and modify line 21 with the path to the newly extended AdvPromptSet dataset.

Note 2: Go to [_base.py](https://github.com/facebookresearch/ResponsibleNLP/blob/main/robbie/datasets/_base.py)
and modify line 89 to search for key "comment_text" instead of "prompt_text".

- Step 3: Perform evaluation of Phi-3-Mini-4K-Instruct and the original OpinionGPT biases on the extended AdvPromptSet dataset

~~~~
python3 evaluate_base.py --gpu-id X
~~~~

## Combined bias evaluation

> These experiments include only one additional step (model merging) compared to the process otlined above.

- Step 1: same as above (skip, if already completed)

- Step 2: same as above (skip, if already completed)

- Step 3: Follow the setup instructions of [MergeKit](https://github.com/arcee-ai/mergekit)

- Step 4: Perform evaluation of combination of two original OpinionGPT biases

~~~~
python3 evaluate_combined_2.py --gpu-id X
~~~~


## Results

> Due to the large size of all produced evaluation files, only two example evaluation files are linked below.

- Phi-3-Mini-4K-Instruct toxicity results can be found [here](https://drive.google.com/file/d/1ISr6FfZUvAT_L6-rKaHQSfV0hK_AMUgN/view?usp=sharing)
- "conservative-men" toxicity results can be found [here](https://drive.google.com/file/d/1SGYIzrOjz1gDr5_5f8GdpzfTwPlB2TJL/view?usp=sharing)




