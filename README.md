# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

## Environment setup
`conda update conda`

`conda env create -f env.yml`

`conda activate devenv`

To remove the environment

`conda remove --name devenv --all`

## Install
`pip install -e .`

The following techniques have been used:

 - Linear regression (Ridge/Lasso)
 - Decision Tree
 - Random Forest

## Training steps
* Edit the config.yml as per the requirements and the guide
* execute model_train.py to train the model
## Scoring steps
* Edit the score_config.yml as per the requirements and the guide
* execute model_score.py to score