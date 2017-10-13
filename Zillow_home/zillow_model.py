import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout
from KerasUtilities import vanilla_nn

train_df = pd.read_csv('train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
test_df = pd.read_csv('sample_submission.csv', low_memory=False)
df = pd.read_csv('properties_2016.csv', low_memory=False)
test_df['parcelid'] = test_df['ParcelId']


def add_date_features(df):
    df['transcation_year'] = df['transactiondate'].dt.year
    df['transcation_month'] = df['transactiondate'].dt.month
    df['transcation_day'] = df['transactiondate'].dt.day
    df['transcation_quarter'] = df['transactiondate'].dt.quarter
    df = df.drop(['transactiondate'], axis=1)
    return df


def clean_columns(df):
    df.columns = map(str.lower, df.columns)
    id_col = [col for col in df.columns if 'id' in col]
    df = df.drop(id_col, axis=1)
    return df


def filter_outliers(df, col):
    upper_lower = df[col].quantile([0.25, 0.75], interpolation='midpoint')
    diff = upper_lower.loc[0.75] - upper_lower.loc[0.25]
    upper = upper_lower.loc[0.75] + 1.5 * diff
    lower = upper_lower.loc[0.25] - 1.5 * diff
    df = df[(df[col] < upper) & (df[col] > lower)]
    return df


def filter_cols_by_nan(df, threshold=0.98):
    ll = len(df)
    nan_cols = [i for i in df.columns if df[i].isnull().sum()*1.0/ll > threshold]
    df = df.drop(nan_cols, axis=1)
    return df


def model_art(X, y):
    model = Sequential()
    import pdb;
    model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metric=['accuracy'])

    model.summary()
    pdb.set_trace()
    checkpointer = ModelCheckpoint(filepath='model.weights.best.h5', verbose=1, save_best_only=True)
    model.fit(X, y, batch_size=256, epochs=50, validation_split=0.2,
                     callbacks=[checkpointer], verbose=1)
    pdb.set_trace()
    return model

train_df = add_date_features(train_df)
train_df = train_df.merge(df, how='left', on='parcelid')
test_df = test_df.merge(df, how='left', on='parcelid')

#train_df = filter_cols_by_nan(train_df)
train_df = clean_columns(train_df)
train_df = train_df .drop(['transcation_year', 'censustractandblock'], axis=1)
train_df = train_df[~train_df.duplicated()]

test_df['transactiondate'] = pd.Timestamp('2016-12-01')
test_df = add_date_features(test_df)
# Fill with large value
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

train_features = ['transcation_month', 'transcation_day', 'transcation_quarter',
                  'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
                  'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
                  'finishedsquarefeet12', 'finishedsquarefeet15', 'finishedsquarefeet50',
                  'fips', 'fireplacecnt', 'fullbathcnt', 'garagecarcnt',
                  'garagetotalsqft', 'hashottuborspa', 'latitude', 'longitude',
                  'lotsizesquarefeet', 'poolcnt', 'rawcensustractandblock', 'roomcnt', 'threequarterbathnbr',
                  'unitcnt', 'yardbuildingsqft17', 'yearbuilt',
                  'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                  'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount']

df = train_df
df['N-life'] = 2018 - df['yearbuilt']
df['A-calculatedfinishedsquarefeet'] = df['finishedsquarefeet12'] + df['finishedsquarefeet15']

# error in calculation of the finished living area of home
df['N-LivingAreaError'] = df['calculatedfinishedsquarefeet'] / df['finishedsquarefeet12']

# proportion of living area
df['N-LivingAreaProp'] = df['calculatedfinishedsquarefeet'] / df['lotsizesquarefeet']
df['N-LivingAreaProp2'] = df['finishedsquarefeet12'] / df['finishedsquarefeet15']

# Amout of extra space
df['N-ExtraSpace'] = df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']
df['N-ExtraSpace-2'] = df['finishedsquarefeet15'] - df['finishedsquarefeet12']

# Total number of rooms
df['N-TotalRooms'] = df['bathroomcnt'] + df['bedroomcnt']

# Average room size
#df['N-AvRoomSize'] = df['calculatedfinishedsquarefeet'] / df['roomcnt']

# Number of Extra rooms
df['N-ExtraRooms'] = df['roomcnt'] - df['N-TotalRooms']

# Ratio of the built structure value to land area
df['N-ValueProp'] = df['structuretaxvaluedollarcnt'] / df['landtaxvaluedollarcnt']

# Does property have a garage, pool or hot tub and AC?
#df['N-GarPoolAC'] = ((df['garagecarcnt'] > 0) & (df['pooltypeid10'] > 0) & (df['airconditioningtypeid'] != 5)) * 1

df["N-location"] = df["latitude"] + df["longitude"]
df["N-location-2"] = df["latitude"] * df["longitude"]
#df["N-location-2round"] = df["N-location-2"].round(-4)

# Ratio of tax of property over parcel
df['N-ValueRatio'] = df['taxvaluedollarcnt'] / df['taxamount']

# TotalTaxScore
df['N-TaxScore'] = df['taxvaluedollarcnt'] * df['taxamount']

# polnomials of tax delinquency year
df["N-taxdelinquencyyear-2"] = df["taxdelinquencyyear"] ** 2
df["N-taxdelinquencyyear-3"] = df["taxdelinquencyyear"] ** 3

# Length of time since unpaid taxes
df['N-live'] = 2018 - df['taxdelinquencyyear']

train_df = filter_outliers(df, 'logerror')
train_df = train_df[[col for col in train_df.columns if 'N' in col] + ['logerror']]

X_train = np.array(train_df, dtype=np.float32)
y_train = np.array(train_df['logerror'], dtype=np.float32)

import pdb; pdb.set_trace()
X_test = np.array(test_df[train_df.columns], dtype=np.float32)

#model, model_history = vanilla_nn(X_train, y_train, 1024, 0.3, 256, 4)
model = model_art(X_train, y_train)
model.load_weights('model.weights.best.h5')
y_pred = model.predict(X_test)
pdb.set_trace()

submission = pd.DataFrame({'ParcelId': test_df['parcelid'].astype(np.int32),
                           '201610': y_pred, '201611': y_pred, '201612': y_pred,
                           '201710': y_pred, '201711': y_pred, '201712': y_pred})
pdb.set_trace()
BUILD_COUNT = 2
submission.to_csv('submission_{}.csv'.format(BUILD_COUNT), float_format='%.4f', index=False)

