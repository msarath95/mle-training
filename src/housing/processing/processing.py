import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def impute(data, num_impute="mean", cat_impute="most_frequent", num_constant=None, cat_constant=None, **kwargs):
    """Impute data based on the method

    Parameters
    ----------
        data: pd.DataFrame
            input data frame to be imputed
        num_impute: str, default mean
            numerical imputation method
        cat_impute: str, default mode
            categorical imputation method
        num_constant: numeric
            numerical constant to use when the num_imputer is constant
        cat_constant: str
            categorical constant to use when the cat_imputer is constant
    
    Returns
    -------
        data: pd.DataFrame
            input data frame to be imputed
        imputer: object
            imputer object
    """
    dtype_dict = data.dtypes
    if num_impute == "constant":
        num_imputer = SimpleImputer(strategy=num_impute, fill_value=num_constant)
    else:
        num_imputer = SimpleImputer(strategy=num_impute)

    if cat_impute == "constant":
        cat_imputer = SimpleImputer(strategy=cat_impute, fill_value=cat_constant)
    else:
        cat_imputer = SimpleImputer(strategy=cat_impute)
    num_cols = list(data.select_dtypes(include=np.number).columns)
    cat_cols = list(data.select_dtypes(exclude=np.number).columns)
    imputer = ColumnTransformer(
        transformers=[
            ("num", num_imputer, num_cols),
            ("cat", cat_imputer, cat_cols),
        ]
    )
    imputer.fit(data)
    data = imputer.transform(data)
    data = pd.DataFrame(data, columns=num_cols + cat_cols)
    data = data.astype(dtype_dict)
    return data, imputer


def impute_transform(data, imputer):
    """Impute transform for test data set

    Parameters
    ----------
        data: pd.DataFrame
            input data frame to be imputed
        imputer: object
            imputer object

    Returns
    -------
        data: pd.DataFrame
            imputed data frame
    """
    cols = data.columns
    dtype_dict = data.dtypes
    data = imputer.transform(data)
    data = pd.DataFrame(data, columns=cols)
    data = data.astype(dtype_dict)
    return data


def generate_features(data):
    """Generates new features

    Parameters
    ----------
        data: pd.DataFrame
            input data frame

    Returns
    -------
        data: pd.DataFrame
            data frame with new features
    """
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    return data
