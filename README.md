# brats
A simple unet in pytorch for brain tumor segmentation.

## Download data

To dowload the datasets run:
```
curl -L -o ./data/brats20-dataset-training-validation.zip\
  https://www.kaggle.com/api/v1/datasets/download/awsaf49/brats20-dataset-training-validation
```

Then unzip the data:
```
unzip brats20-dataset-training-validation.zip "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0[0-5][0-9]/*"
```

# How to run

python3 -m venv .venv

source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit opencv-python matplotlib nibabel