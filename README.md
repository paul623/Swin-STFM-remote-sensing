# SwinSTFM
Code of [SwinSTFM: Remote Sensing Spatiotemporal Fusion using Swin Transformer](https://ieeexplore.ieee.org/abstract/document/9795183)
# 修改说明
train是可以跑通的，效果很好，很草率的设置了双显卡跑（两张2080ti）

# Environment
`pip install -r requirements.txt`

# Dataset Directory 
`datasets/dataset_directory.png`

# Generating Training Set
`python datasets/generate_data.py`

# Traing
`python train.py`

# Testing
`python test.py`

You could download the model pretrained on LGC from here: https://drive.google.com/file/d/1GkwhIzIPUl3q85NKZMLW3jW4DxqG_GSG/view?usp=sharing.
