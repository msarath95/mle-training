import unittest

import numpy as np
import pandas as pd
from housing.processing import processing as pr


class TestHousing(unittest.TestCase):
    def test_impute(self):
        data = pd.DataFrame(np.random.random((1000, 5)), columns=[
            "x{}".format(x) for x in range(5)])
        data["x5"] = np.random.choice(["A", "B", "C", "D", "E"], 1000)
        data["target"] = 2 * data["x0"] + 3 * data["x4"]
        data.loc[0:200, ["target", "x5"]] = np.nan
        try:

            impute_by_mean, imputed_mean_value = pr.impute(
                data, num_impute="mean", cat_impute='most_frequent')
            assert impute_by_mean.isnull().any().sum().sum() == 0

            impute_by_median, imputed_median_value = pr.impute(
                data, num_impute="median", cat_impute='most_frequent')
            assert impute_by_median.isnull().any().sum().sum() == 0

            impute_by_mode, imputed_mode_value = pr.impute(
                data, num_impute="most_frequent", cat_impute='most_frequent')
            assert impute_by_mode.isnull().any().sum().sum() == 0
        except AssertionError as err:
            print(err)
