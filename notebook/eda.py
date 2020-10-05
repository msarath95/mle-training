import matplotlib.pyplot as plt
from housing.preparation import data_utils as du
from housing.preparation import utils as ut

ut.configure_logger()
cfg_path = "./config/config.yml"
cfg = ut.read_config(cfg_path)

train, test = du.prepare_model_data(cfg)

# Target
train["median_house_value"].describe()
train["median_house_value"].hist()
plt.show()


train.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = train.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
