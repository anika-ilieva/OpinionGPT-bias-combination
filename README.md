# Exploring Bias Combinations in Large Language Models

> Source code of Master's Thesis

<img width="657" height="222" alt="method" src="https://github.com/user-attachments/assets/5778f737-9efe-45a7-9118-c18b4383336a" />


- Step 1 (A): [MergeKit Merging Library](https://github.com/arcee-ai/mergekit) is used within this project.

- Step 2 (B): Linear merging is selected as the optimal method for bias combiantion.
Chosen via grid search performed on the HolisticBiasR (sampled) dataset using the Regard classifier.
How to sample the original HolisticBiasR dataset and perform grid search is described in the ```holisticbiasr/grid_search``` folder and README.

- Step 3 (C): Extend AdvPromptSet to enable evaluation of bias intersection including political demographics.
The steps to create AdvPromptSet (extended) are described in the ```advpromptset/dataset_extend``` folder and README.

- Step 4: Evaluate Phi-3-Mini-4K-Instruct, the original OpinionGPT biases and the two-, three-, and four-bias combinations on the full-scale HolisticBiasR dataset using the Regard classifier. Detailed instructions can be found in ```holisticbiasr/evaluate```.
  
- Step 5: Evaluate Phi-3-Mini-4K-Instruct, the original OpinionGPT biases and the two-bias combinations on the extended AdvPromptSet dataset using the ToxiGen classifier. Detailed instructions can be found in ```advpromptset/evaluate```.

## Results 

- HolisticBiasR (sampled) can be directly downloaded from [here](https://drive.google.com/drive/folders/1et_UAKGWt7VRRhU3_vGqCPux0L4-MDYv?usp=sharing) or from [HuggingFace](https://huggingface.co/datasets/anika-ilieva/HolisticBiasR-sampled).
- AdvPromptSet (extended) can be directly downloaded from [here](https://drive.google.com/drive/folders/1et_UAKGWt7VRRhU3_vGqCPux0L4-MDYv?usp=sharing) or from [HuggingFace](https://huggingface.co/datasets/anika-ilieva/AdvPromptSet-extended).
- Examples of the evaluation results of each step can be found [here](https://drive.google.com/drive/folders/1et_UAKGWt7VRRhU3_vGqCPux0L4-MDYv?usp=sharing).

## Useful Links

- [OpinionGPT Bias Adapter Collection](https://huggingface.co/collections/HU-Berlin-ML-Internal/opiniongpt-adapters-66f404e650552022cd6b0353)
- [Base Model](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct)
- [MergeKit Merging Library](https://github.com/arcee-ai/mergekit)
- [ResponsibleNLP repository](https://github.com/facebookresearch/ResponsibleNLP) 
