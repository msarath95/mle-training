import matplotlib.pyplot as plt

from housing.modeling import train as train
from housing.preparation import data_utils as du
from housing.preparation import utils as ut

cfg_path = "./config/config.yml"
cfg = ut.read_config(cfg_path)

score_cfg_path = "./config/score_config.yml"
score_cfg = ut.read_config(score_cfg_path)

train, test = du.prepare_model_data(cfg)
test = pd.read_csv()
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]
model = train.model_selection_fit(cfg, X, y)
