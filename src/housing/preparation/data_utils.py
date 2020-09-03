import os
import pickle as pkl
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from housing.processing import processing as pr


def is_data_exists(housing_path):
    """This function is used to check if the data exist in the given path
    Parameters
    ----------
        housing_path : str,
            Path to download the data to.
    """
    return os.path.exists(housing_path)


def fetch_housing_data(housing_url, housing_path, over_write_raw_data=False, **kwargs):
    """This function is used to download the data
    Parameters
    ----------
        housing_url : str,
            The URL path to download the data from.
        housing_path : str,
            Path to download the data to.
        over_write_raw_data : bool, default True
            To over_write the existing data
    """
    create_data = not is_data_exists(housing_path) or over_write_raw_data
    if create_data:
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()


def load_housing_data(housing_path):
    """This function is used to read the data
    Parameters
    ----------
        housing_path : str,
            file path.
    Return:
    -------
        data set
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def get_train_test_split(data, sampling_method="stratified", seed=42, test_size=0.2):
    """This function is to split the train and test data"""
    if sampling_method == "stratified":
        data["income_cat"] = pd.cut(
            data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
        )
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for train_index, test_index in split.split(data, data["income_cat"]):
            train = data.loc[train_index]
            test = data.loc[test_index]
        train.drop("income_cat", axis=1, inplace=True)
        test.drop("income_cat", axis=1, inplace=True)
    elif sampling_method == "random":
        train, test = train_test_split(data, test_size=test_size, random_state=seed)
    return train, test


def prepare_model_data(cfg):
    """This function creates the train and test model data
    Parameters
    ----------
        cfg: dict
    Returns:
    --------
        void
    """
    create_data = not is_data_exists(cfg["model_data_path"] + "/train.csv") or cfg["over_write_model_data"]
    fetch_housing_data(**cfg)
    data = load_housing_data(cfg["housing_path"])
    if create_data:
        train, test = get_train_test_split(data, cfg["sampling_method"], cfg["seed"], cfg["test_size"])

        # Impute
        train, imputer = pr.impute(train, **cfg)
        test = pr.impute_transform(test, imputer)
        pkl.dump(imputer, open(cfg["models_path"] + "/imputer.pkl", "wb"))
        # Feature Engineer
        train = pr.generate_features(train)
        test = pr.generate_features(test)

        # one-hotencode with column names?

        train.to_csv(cfg["model_data_path"] + "/train.csv", index=False)
        test.to_csv(cfg["model_data_path"] + "/test.csv", index=False)
    train = pd.read_csv(cfg["model_data_path"] + "/train.csv")
    test = pd.read_csv(cfg["model_data_path"] + "/test.csv")
    return train, test
