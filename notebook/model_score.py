import pandas as pd

from housing.modeling import score as sr

score_cfg_path = "./config/score_config.yml"
score_cfg = ut.read_config(score_cfg_path)
score_df = pd.read_csv(score_cfg["score_data_path"])
try:
    X = score_df.drop("median_house_value", axis=1)
    y = score_df["median_house_value"]
except:
    X = score_df
y_hat = sr.score(score_cfg, X, preproc=score_cfg["preproc"])
