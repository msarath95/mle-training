import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def impute(data, num_impute="mean", cat_impute="mode", num_constant=None, cat_constant=None, **kwargs):
    """Impute data based on the method
    Parameters:
    -----------
        data: data frame
        num_impute: str numerical imputation method
        cat_impute: str categorical imputation method
        num_constant: numerical constant to use when the num_imputer is constant
        cat_constant: categorical constant to use when the cat_imputer is constant
    Returns:
    --------
        data: data frame
        imputer: imputer object
    """
    cols = data.columns
    dtype_dict = data.dtypes
    if num_impute == "constant":
        num_imputer = SimpleImputer(strategy=num_impute, fill_value=num_constant)
    else:
        num_imputer = SimpleImputer(strategy=num_impute)

    if cat_impute == "constant":
        cat_imputer = SimpleImputer(strategy=cat_impute, fill_value=cat_constant)
    else:
        cat_imputer = SimpleImputer(strategy=cat_impute)
    imputer = ColumnTransformer(
        transformers=[
            ("num", num_imputer, data.select_dtypes(include=np.number).columns),
            ("cat", cat_imputer, data.select_dtypes(exclude=np.number).columns),
        ]
    )
    imputer.fit(data)
    data = imputer.transform(data)
    data = pd.DataFrame(data, columns=cols, dtype=dtype_dict.values)
    return data, imputer


def impute_transform(data, imputer):
    """Impute transform for test data set
    Parameters:
    -----------
        data: data frame
        imputer: imputer object
    Returns:
    --------
        data: imputed data frame
    """
    cols = data.columns
    dtype_dict = data.dtypes
    data = imputer.transform(data)
    data = pd.DataFrame(data, columns=cols, dtype=dtype_dict.values)
    return data


def generate_features(data):
    """Generates new features
    Parameters:
    -----------
        data: data frame
    Returns:
    --------
        data: with new features
    """
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    return data
