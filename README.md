# Proposed MDB-Net User Guide

## Group assignment repo

github\CSCI935-CV

    src\train_multitask.py

    src\train_resnet18.py

    src\make_masks.py

    src\seg_leaf.py


## Data Source

github\CSCI935-CV\Dhan-Shomadhan


## CMD

### Environment

conda activate 935cv

cd ~/github/CSCI935-CV/src

### Mask Generation

python src/make_masks.py

### Model Training and Evaluation

python src/train_multitask.py --field_train cls


## Log files

### Masks

github/CSCI935-CV/Dhan-Shomadhan/_interim/


### Experimental Result
github/Dhan-Shomadhan/_interim/runs/mt_mask_swap0.5_seg1_field-cls
