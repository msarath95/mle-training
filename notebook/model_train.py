from housing.modeling import eval as ev
from housing.modeling import score as sr
from housing.modeling import train as tr
from housing.preparation import data_utils as du
from housing.preparation import utils as ut

cfg_path = "./config/config.yml"
cfg = ut.read_config(cfg_path)

train, test = du.prepare_model_data(cfg)
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]
model = tr.model_selection_fit(cfg, X, y)
y_train_hat = sr.score(cfg, X, preproc=True)
y_test_hat = sr.score(cfg, test.drop("median_house_value", axis=1), preproc=True)

train_performance = ev.get_performance(y, y_train_hat)
test_performance = ev.get_performance(test["median_house_value"], y_test_hat)
