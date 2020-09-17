import pickle as pkl

from housing.preparation import data_utils as du


def score(cfg, X, preproc=False):
    """Based on the input from config the data will be scored

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
    imputer = pkl.load(open(cfg["models_path"] + "/imputer_{version}.pkl".format(**cfg), "rb"))
    model = pkl.load(open(cfg["models_path"] + "/model_{version}.pkl".format(**cfg), "rb"))
    if not preproc:
        X = du.prepare_test_data(X, imputer)
    y_hat = model.predict(X)
    return y_hat
