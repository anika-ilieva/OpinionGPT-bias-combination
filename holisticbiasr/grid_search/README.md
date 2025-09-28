# Grid Search

> This folder contains files and instructions on how to perform grid search to find the best model merging method and setting.

<img width="746" height="301" alt="grid_search" src="https://github.com/user-attachments/assets/863fc0be-68c4-4280-904e-7d4c99d8cd82" />

## Create HolisticBiasR (sampled)

- Step 1: Follow the setup instructions in the [ResponsibleNLP repository](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias) to enable the original HolisticBiasR dataset generation.

- Step 2: Navigate to [sentences.py](https://github.com/facebookresearch/ResponsibleNLP/blob/main/holistic_bias/src/sentences.py) and perform the changes described in [sentences.txt](https://github.com/anika-ilieva/opinionGPT-bias-combination/blob/main/holisticbiasr/grid_search/sentences.txt).

- Step 3: With the updated sentences.py, create the sampled version of HolisticBiasR

~~~~
python -m holistic_bias.generate_sentences --use-small-set --dataset-version v1.1 ./hbr_sampled/
~~~~

## Evaluate original OpinionGPT biases on HolisticBiasR (sampled)

- Step 1: Setup the [ResponsibleNLP repository](https://github.com/facebookresearch/ResponsibleNLP/tree/main), if not already done

- Step 2: Peform Step 2 from above, if not already done

- Step 3: Update the out_dir path in line 28 of [holisticbiasr.py](https://github.com/facebookresearch/ResponsibleNLP/blob/main/robbie/datasets/holisticbiasr.py) to "noun_phrases__small_set.csv" instead of "noun_phrases.csv"

- Step 4: Run evaluate_base.py script

~~~~
python3 evaluate_base.py
~~~~

## Evaluate combined OpinionGPT biases on HolisticBiasR (sampled)

- Step 1: Setup the [ResponsibleNLP repository](https://github.com/facebookresearch/ResponsibleNLP/tree/main), if not already done

- Step 2: Peform Step 2 from above, if not already done

- Step 3: Follow the setup instructions of [MergeKit](https://github.com/arcee-ai/mergekit), if not already done

- Step 4: Perform grid search using grid_search_combined.py

~~~~
python3 grid_search_combined.py
~~~~

## Results 

> Due to the large number of all produced evaluation files, only two example evaluation files on HolisticBiasR (sampled) are linked below.

- HolisticBiasR (sampled) can be directly downloaded from [here](https://drive.google.com/drive/folders/1roQJ1SnxdNDTNBP9zx95c-eULUOj5JLP?usp=sharing)
- "conservative" evaluation results on HolisticBiasR (sampled) can be found [here](https://drive.google.com/file/d/1XaHZ4K54aTKeKf7ZUJTjXV_I7xZ47vJU/view?usp=sharing)
- "conservative-men" (method=breadcrumbs, weight=[0.3,0.3], lambda=1, density=0.5, gamma=0.1) results on HolisticBiasR (sampled) can be found [here](https://drive.google.com/file/d/1_92hIjU0mE7Af90I4E5m2fHOZN56QaFU/view?usp=sharing)
