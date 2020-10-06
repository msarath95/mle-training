import logging
import os
import pickle as pkl

from housing.preparation import data_utils

logger = logging.getLogger(__name__)


def score(cfg, X, preproc=False):
    """Based on the input from config the data will be scored.

    Parameters
    ----------
        cfg: dict
            configuration dict
        X: pd.DataFrame
            input data
        preproc: bool
            to do preprocessing
    Return
    ------
        y_hat: np.array
            predictions
    """
    logger.info("no of obeservation in data {}".format(X.shape[0]))
    logger.info("scoring with {}".format(cfg["version"]))
    imputer = pkl.load(open(os.path.join(cfg["models_path"], "imputer_{version}.pkl".format(**cfg)), "rb"))
    model = pkl.load(open(os.path.join(cfg["models_path"], "model_{version}.pkl".format(**cfg)), "rb"))
    if not preproc:
        X = data_utils.prepare_test_data(X, imputer)
    y_hat = model.predict(X)
    return y_hat
