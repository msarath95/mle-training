import matplotlib.pyplot as plt

from housing.modeling import train as train
from housing.preparation import data_utils as du
from housing.preparation import utils as ut

cfg_path = "./config/config.yml"
cfg = ut.read_config(cfg_path)

train, test = du.prepare_model_data(cfg)
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]
model = train.model_selection_fit(cfg, X, y)
