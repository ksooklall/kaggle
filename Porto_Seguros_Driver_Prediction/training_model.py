import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import clean_columns, standerize

from keras import initializers
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout

BUILD_COUNT = 3

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df[train_df == -1] = np.nan
train_df = clean_columns(train_df)
import pdb;

# Nan columns kept and replaced
nan_train_df = train_df.fillna(-999)
# Nan columns removed
full_cols = train_df.columns[~(train_df.isnull().sum() > 0)]
full_train_df = train_df[full_cols]

split = int(len(train_df) * 0.8)
mean, std, std_train_df = standerize(full_train_df.drop(['target'], axis=1))


def get_training_testing(df):
    if 'target' not in df.columns:
        df['target'] = train_df['target']

    X_train = np.array(df.drop(['target'], axis=1)).astype(np.float32)
    y_train = np_utils.to_categorical(df['target'], 2)
    return X_train, y_train

X, y, = get_training_testing(full_train_df)
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.25)
X_train, X_valid, y_train, y_valid = X_train_valid[:split], X_train_valid[split:], y_train_valid[:split], y_train_valid[split:]
X_valid, X_test, y_valid, y_test = X_valid * std + mean, X_test * std + mean, y_valid * std + mean, y_test * std + mean

trunc_norm = initializers.truncated_normal(stddev=0.01)

# Hyperparameters
hidden_units = 1024
drop_out = 0.3
batch_size = 512

def nn_model(hidden_units, drop_out, batch_size, depth):
    model = Sequential()

    model.add(Dense(hidden_units, kernel_initializer=trunc_norm, activation='relu', input_shape=(X_train.shape[1],)))

    for i in range(depth):
        model.add(Dense(hidden_units//2, kernel_initializer=trunc_norm, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_out))

        if hidden_units < 6:
            hidden_units = 8

    model.add(Dense(hidden_units//2, kernel_initializer=trunc_norm, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=30, validation_data=(X_valid, y_valid), callbacks=[checkpointer], verbose=1)
    #0.1535
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Model acc: {}'.format(score[1] * 100))

    return model


def predict_and_submit(model):
    x_test = np.array(test_df[full_cols[1:]])
    y_pred = model.predict(x_test)
    submission = pd.DataFrame({'id': test_df['id'], 'target': y_pred[:, 1]})
    submission.to_csv('submission_{}.csv'.format(BUILD_COUNT), index=False)

#submit(model)
nn_model(hidden_units, drop_out, batch_size, 2)
pdb.set_trace()
