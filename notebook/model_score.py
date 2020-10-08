import pandas as pd
from housing.modeling import score as sr
from housing.preparation import utils as ut

score_cfg_path = "./config/score_config.yml"
score_cfg = ut.read_config(score_cfg_path)
score_df = pd.read_csv(score_cfg["score_data_path"])
if "median_house_value" in score_df.columns:
    X = score_df.drop("median_house_value", axis=1)
    y = score_df["median_house_value"]
else:
    X = score_df
y_hat = sr.score(score_cfg, X, preproc=score_cfg["preproc"])
