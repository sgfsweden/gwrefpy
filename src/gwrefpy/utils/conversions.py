import pandas as pd


def datetime_to_float(date_time: pd.Timestamp) -> float:
    """
    Convert a datetime object to a float representation.

    Parameters
    ----------
    date_time : datetime
        The datetime object to convert.

    Returns
    -------
    float
        The float representation of the datetime.
    """
    return date_time.timestamp()


def float_to_datetime(float_time: float) -> pd.Timestamp:
    """
    Convert a float representation of time back to a datetime object.

    Parameters
    ----------
    float_time : float
        The float representation of time.

    Returns
    -------
    pd.Timestamp
        The corresponding datetime object.
    """
    return pd.to_datetime(float_time, unit="s")
