import os
import pandas as pd


def read_pickled_data(path_items):
    """Return pickled dataframe from path defined by items"""
    return pd.read_pickle(os.path.join(*path_items))


def read_feathered_data(path_items):
    """Return pickled dataframe from path defined by items"""
    return pd.read_feather(os.path.join(*path_items))
