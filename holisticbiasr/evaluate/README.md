# HolisticBiasR Evaluation

> Evaluate Phi-3-Mini-4K-Instruct, the original OpinionGPT biases, and combinations of two, three, and four OpinionGPT biases using the full-scale AdvPromptSet dataset and the ToxiGen classifier.

- Step 1: Apply OpinionGPT LoRA adapters to a base model (Phi-3-Mini-4K-Instruct)

Navigate to utils and execute [save_models.py](https://github.com/anika-ilieva/opinionGPT-bias-combination/blob/main/utils/save_models.py)

~~~~
python3 save_models.py
~~~~

- Step 2: Follow the setup instructions of [ResponsibleNLP](https://github.com/facebookresearch/ResponsibleNLP/tree/main)

- Step 3: Follow the setup instructions of [MergeKit](https://github.com/arcee-ai/mergekit)

## Base Model Evaluation

- Step 4: Perform evaluation of Phi-3-Mini-4K-Instruct and all original OpinionGPT biases on the full-scale HolisticBiasR dataset

~~~~
python3 evaluate_base.py --gpu-id X
~~~~

## Two-Bias Combination Evaluation

- Step 5: Perform evaluation of all two-bias combinations of the original OpinionGPT biases on the full-scale HolisticBiasR dataset

~~~~
python3 evaluate_combined_2.py --gpu-id X
~~~~

## Three-Bias Combination Evaluation

- Step 6: Perform evaluation of all three-bias combinations of the original OpinionGPT biases on the full-scale HolisticBiasR dataset

~~~~
python3 evaluate_combined_3.py --gpu-id X
~~~~

## Four-Bias Combination Evaluation 

- Step 7: Perform evaluation of all four-bias combinations of the original OpinionGPT biases on the full-scale HolisticBiasR dataset
  
~~~~
python3 evaluate_combined_4.py --gpu-id X
~~~~

## Results 

> Due to the large number and size of all produced evaluation files, only a few example evaluation files are linked below.

- "conservative" evaluation results on the full-scale HolisticBiasR dataset can be found [here](https://drive.google.com/file/d/1fzcpc-YN0UIKL3KemIL_Knxfl_aBdD46/view?usp=sharing)
- "conservative-men" evaluation results on the full-scale HolisticBiasR dataset can be found [here](https://drive.google.com/file/d/1jTE2_bIZtroVmDm7BfSbTjV5O7uqVIJY/view?usp=sharing)
- "conservative-men-american" evaluation results on the full-scale HolisticBiasR dataset can be found [here](https://drive.google.com/file/d/1l5Yv3zXyqG7KzrnKFM4kr6Xc59M1ZbzY/view?usp=sharing)
- "conservative-men-american-old_people" evaluation results on the full-scale HolisticBiasR dataset can be found [here](https://drive.google.com/file/d/1YHGULFP96XohHA9DWtmcuA92QvYwQN0g/view?usp=sharing)
