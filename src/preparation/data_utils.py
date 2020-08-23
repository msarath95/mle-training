import os
import tarfile
from six.moves import urllib
import pandas as pd


def is_data_exists(housing_path):
    """This function is used to check if the data exist in the given path
    Parameters
    ----------
        housing_path : str,
            Path to download the data to.
    """
    return os.path.exists(housing_path)


def fetch_housing_data(housing_url, housing_path, over_write=True):
    """This function is used to download the data
    Parameters
    ----------
        housing_url : str,
            The URL path to download the data from.
        housing_path : str,
            Path to download the data to.
        over_write : bool, default True
            To over_write the existing data
    """
    create_data = not is_data_exists(housing_path) or over_write
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
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
