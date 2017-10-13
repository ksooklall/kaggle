import numpy as np
import pandas as pd

def clean_columns(df, remove_id=True):
    """
    Lower and strips all column names
    :param df: dataframe
    :return: dateframe
    """
    df.columns = map(str.lower, df.columns)
    if remove_id:
        id_col = [col for col in df.columns if 'id' in col]
        df = df.drop(id_col, axis=1)
    return df


def filter_cols_by_nan(df, threshold=0.98):
    """
    Remove cols with too many NaNs based on a threshold
    :param df: Dataframe
    :param threshold: Float
    :return: Dataframe
    """
    row_count = len(df)
    nan_cols = [col for col in df.columns if df[col].isnull().sum()*1.0/row_count > threshold]
    df = df.drop(nan_cols, axis=1)
    print('Removed: {}'.format(nan_cols))
    return df


def standerize(df, use_cols=[]):
    """
    Standeries all values in a dataframe
    If cols is given, will only standerize values in given column
    :param df: Dataframe
    :param cols: List of columns
    :return:
    """
    cols = df.columns
    if use_cols:
        cols = use_cols

    mu = df[cols].mean()
    std = df[cols].std()
    standerized_df = (df[cols] - mu) / std

    return mu, std, standerized_df


def down_sampling(df, seed):
    np.random.seed(seed)
    positive = df['target'] == 0
    positive_claims = df[positive]
    negative_claims = df.loc[np.random.choice(df[~positive].index, len(positive_claims))]
    balanced_df = pd.concat([positive_claims, negative_claims], axis=0).reset_index(drop=True)
    return balanced_df


def up_sampling(df, seed):
    np.random.seed(seed)
    negative = df['target'] == 1
    negative_claims = df[negative]
    positive_claims = df.loc[np.random.choice(df[~negative].index, len(negative_claims))]
    balanced_df = pd.concat([positive_claims, negative_claims], axis=0).reset_index(drop=True)
    return balanced_df