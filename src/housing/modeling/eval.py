import numpy as np
import pandas as pd
from sklearn import metrics


def reg_metric_MAPE(y_true, y_hat):
    """
    Computes Mean Absolute Percent error
    Ignore values with y_true=0.

    Parameters
    ----------
        y_true: np.array
            actual values
        y_hat: np.array
            predicted values

    Return
    ------
        mape: float
            MAPE
    """
    if len(y_true) != len(y_hat):
        raise ValueError("y_true & y_hat should be of equal length")
    y_true, y_hat = np.array(y_true), np.array(y_hat)
    mape = np.mean(np.abs((y_true-y_hat)/y_true))
    return mape


def reg_metric_WMAPE(y_true, y_hat):
    """Computes Weighted Mean Absolute Deviation. Similar to reg_metric_MAPE but weighted by y_true.
    This ignores y_true=0 and tends to be higher if the errors are higher for higher y_true values.

    Parameters
    ----------
        y_true: np.array
            actual values
        y_hat: np.array
            predicted values

    Return
    ------
        wmape: float
            WMAPE
    """
    if len(y_true) != len(y_hat):
        raise ValueError("y_true & y_hat should be of equal length")
    y_true, y_hat = np.array(y_true), np.array(y_hat)
    return np.sum(np.abs(y_true-y_hat))/sum(y_true)


def reg_metric_RMSE(y_true, y_hat):
    """This function is to compute Root Mean Squared Error.

    Parameters
    ----------
        y_true: np.array
            actual values
        y_hat: np.array
            predicted values

    Return
    ------
        rmse: float
            RMSE

    """
    if len(y_true) != len(y_hat):
        raise ValueError("y_true & y_hat should be of equal length")
    y_true, y_hat = np.array(y_true), np.array(y_hat)
    return np.sqrt(np.mean(np.square(y_true-y_hat)))


def get_performance(y_true, y_hat):
    """
    This function computes the evaluation metrics for regression.

    Parameters
    ----------
        y_true: np.array
            actual values
        y_hat: np.array
            predicted values

    Return
    ------
        out_metric: dict
            performance metrics
    """
    if len(y_true) != len(y_hat):
        raise ValueError("y_true & y_hat should be of equal length")

    df = pd.DataFrame.from_dict({'y': y_true, 'y_hat': y_hat})
    out_metric = {}
    out_metric['r2_score'] = metrics.r2_score(df['y'], df['y_hat'])
    out_metric['MAD'] = metrics.median_absolute_error(df['y'],
                                                      df['y_hat'])
    out_metric['MAPE'] = reg_metric_MAPE(df['y'], df['y_hat'])
    out_metric['WMAPE'] = reg_metric_WMAPE(df['y'], df['y_hat'])
    out_metric['RMSE'] = reg_metric_RMSE(df['y'], df['y_hat'])
    return out_metric
