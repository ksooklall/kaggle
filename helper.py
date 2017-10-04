
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

