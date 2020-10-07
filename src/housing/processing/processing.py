import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.utils.validation import check_array, check_is_fitted

logger = logging.getLogger(__name__)


class Imputer(_BaseImputer, TransformerMixin):
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
    """
    def __init__(self, num_impute="mean", cat_impute="most_frequent", num_constant=None,
                 cat_constant=None, missing_values=np.nan, add_indicator=False):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.num_constant = num_constant
        self.cat_impute = cat_impute
        self.cat_constant = cat_constant

    def fit(self, X, y=None):
        check_array(X, accept_large_sparse=False, dtype=object, force_all_finite="allow-nan")
        self.dtype_dict_ = X.dtypes
        if self.num_impute == "constant":
            self.num_imputer_ = SimpleImputer(strategy=self.num_impute, fill_value=self.num_constant)
        else:
            self.num_imputer_ = SimpleImputer(strategy=self.num_impute)

        if self.cat_impute == "constant":
            self.cat_imputer_ = SimpleImputer(strategy=self.cat_impute, fill_value=self.cat_constant)
        else:
            self.cat_imputer_ = SimpleImputer(strategy=self.cat_impute)
        self.num_cols_ = list(X.select_dtypes(include=np.number).columns)
        self.cat_cols_ = list(X.select_dtypes(exclude=np.number).columns)
        self.imputer_ = ColumnTransformer(
            transformers=[
                ("num", self.num_imputer_, self.num_cols_),
                ("cat", self.cat_imputer_, self.cat_cols_),
            ]
        )
        self.imputer_.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "is_fitted_")
        X = self.imputer_.transform(X)
        X = pd.DataFrame(X, columns=self.num_cols_ + self.cat_cols_)
        X = X.astype(self.dtype_dict_)
        return X

    def _more_tags(self):
        return {"allow_nan": True, "X_types": ["2darray", "string"]}


def impute(data, num_impute="mean", cat_impute="most_frequent", num_constant=None, cat_constant=None, **kwargs):
    """Impute data based on the method.

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
    logger.info("INFO: numerical columns imputation: {}".format(num_cols))
    logger.info("INFO: cateogrical columns imputation: {}".format(cat_cols))
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
    """Impute transform for test data set.

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


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        self._cols = list(X.columns)
        self._rooms_ix = self._cols.index("total_rooms")
        self._bedrooms_ix = self._cols.index("total_bedrooms")
        self._population_ix = self._cols.index("population")
        self._household_ix = self._cols.index("households")
        if self.add_bedrooms_per_room:
            self._cols += ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        else:
            self._cols += ["rooms_per_household", "population_per_household"]
        return self

    def transform(self, X, y=None):
        X = X.values
        rooms_per_household = X[:, self._rooms_ix] / X[:, self._household_ix]
        population_per_household = X[:, self._population_ix] / X[:, self._household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self._bedrooms_ix] / X[:, self._rooms_ix]
            return pd.DataFrame(
                np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room],
                columns=self._cols)
        else:
            return pd.DataFrame(np.c_[X, rooms_per_household, population_per_household], columns=self._cols)

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)


def generate_features(data):
    """Generates new features.

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
