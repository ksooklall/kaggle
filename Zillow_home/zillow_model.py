import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout


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
    checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)
    model.fit(X, y, batch_size=256, epochs=100, validation_split=0.2,
                     callbacks=[checkpointer], shuffle=True, verbose=1)
    pdb.set_trace()
    return model

train_df = add_date_features(train_df)
train_df = train_df.merge(df, how='left', on='parcelid')
test_df = test_df.merge(df, how='left', on='parcelid')

train_df = filter_cols_by_nan(train_df)
train_df = clean_columns(train_df)
train_df = train_df .drop(['transcation_year', 'censustractandblock'], axis=1)
train_df = train_df[~train_df.duplicated()]

test_df['transactiondate'] = pd.Timestamp('2016-12-01')
test_df = add_date_features(test_df)
# Fill with large value
train_df = train_df.fillna(-9999)
test_df = test_df.fillna(-9999)

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

X_train = np.array(train_df[train_features], dtype=np.float32)
y_train = np.array(train_df['logerror'], dtype=np.float32)

X_test = np.array(test_df[train_features], dtype=np.float32)

import pdb; pdb.set_trace()
model = model_art(X_train, y_train)
model.load_weights('model.weights.best')
y_pred = model.predict(X_test)
pdb.set_trace()

submission = pd.DataFrame({'ParcelId': test_df['parcelid'].astype(np.int32),
                           '201610': y_pred, '201611': y_pred, '201612': y_pred,
                           '201710': y_pred, '201711': y_pred, '201712': y_pred})
pdb.set_trace()
submission_indx = 1
submission.to_csv('submission_{}03d.csv'.format(submission_indx), float_format='%.4f', index=False)

