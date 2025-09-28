# AdvPromptSet Extension 

> Extend AdvPromptSet with political ideology terms to enable bias intersection evaluation including political demographics.

## Overview

<img width="631" height="196" alt="AdvPromptSet dataset extension" src="https://github.com/user-attachments/assets/74110ff9-efc7-46b3-ad74-0569f085ea99" />

- Step 0: Define initial word list (manually)

- Step 1: Expand word list via Sentence-BERT

- Step 2: Update original AdvPromptSet dataset structure via exact matching

## Usage

### Generate original AdvPromptSet dataset locally 

Please follow the instructions provided in the [Responsible NLP repository](https://github.com/facebookresearch/ResponsibleNLP/tree/main/AdvPromptSet).

### Create word lists

~~~~
python3 create_word_lists.py
~~~~

### Update original AdvPromptSet structure

Please adjust the input and output paths to match your local setup. 
~~~~
python3 dataset_extend.py
~~~~

## Results 

- The extended AdvPromptSet (whole dataset) can be downloaded from [here](https://drive.google.com/file/d/1SHXIK8QFUx4VsKrej2sTAWz7xrSKMab4/view?usp=sharing)
- The extended AdvPromptSet (10k samples) can be downloaded from [here](https://drive.google.com/file/d/1SHXIK8QFUx4VsKrej2sTAWz7xrSKMab4/view?usp=sharing)
