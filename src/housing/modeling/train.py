import logging
import os
import pickle as pkl

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


def model_selection_fit(cfg, X, y):
    """Based on the input from config the model will be selected

    Parameters
    ----------
        cfg: dict
            configuration

    Return
    ------
        model: object
            sklearn model object
    """
    logger.info("no of obeservation in data {}".format(X.shape[0]))
    logger.info("training {}".format(cfg["algo"]))
    if cfg["algo"] == "linear-ridge":
        model_cfg = cfg["linear-ridge"]
        model_cfg["random_state"] = cfg["seed"]
        model = linear_model.Ridge(**model_cfg)
    elif cfg["algo"] == "linear-lasso":
        model_cfg = cfg["linear-lasso"]
        model_cfg["random_state"] = cfg["seed"]
        model = linear_model.Lasso(**model_cfg)
    elif cfg["algo"] == "decision_tree":
        model_cfg = cfg["decision_tree"]
        model_cfg["random_state"] = cfg["seed"]
        model = DecisionTreeRegressor(**model_cfg)
    elif cfg["algo"] == "random_forest":
        model_cfg = cfg["random_forest"]
        model_cfg["random_state"] = cfg["seed"]
        model = RandomForestRegressor(**model_cfg)
    model.fit(X, y)
    pkl.dump(model, open(os.path.join(cfg["models_path"], "model_{version}.pkl".format(**cfg)), "wb"))
    return model
