# brats
A simple unet in pytorch for brain tumor segmentation.

## Clone me
```
git clone https://github.com/Lucien-MG/brats
```

then enter in the repo:
```
cd brats
```


## Download data

To dowload the datasets run:
```
curl --create-dirs -L -o ./data/brats20-dataset-training-validation.zip https://www.kaggle.com/api/v1/datasets/download/awsaf49/brats20-dataset-training-validation
```

Then unzip the data:
```
unzip data/brats20-dataset-training-validation.zip "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_0[0-5][0-9]/*" -d ./data
```

## How to run

Install deps
```
python3 -m venv .venv

source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit opencv-python matplotlib nibabel
```

Run app
```
```