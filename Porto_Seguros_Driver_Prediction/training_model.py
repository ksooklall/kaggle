import pandas as pd
import numpy as np

from helper import clean_columns, filter_cols_by_nan

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import np_utils

BUILD_COUNT = 1

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Notes
# No duplicates
# No nans
train_df[train_df == -1] = np.nan
train_df = clean_columns(train_df)

# Nan columns kept and replaced
nan_train_df = train_df.fillna(-999)
# Nan columns removed
full_cols = train_df.columns[~(train_df.isnull().sum() > 0)]
full_train_df = train_df[full_cols]

# One hot incode targets
import pdb; pdb.set_trace()
targets = full_train_df['target']

X = np.array(full_train_df.drop(['target'], axis=1)).astype(np.float32)
#y = np.eye(2)[targets]
y = np_utils.to_categorical(targets, 2)

pdb.set_trace()
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)
pdb.set_trace()
model.fit(X, y, batch_size=512, epochs=5, validation_split=0.2, callbacks=[checkpointer], verbose=1)

pdb.set_trace()
x_test = test_df[full_cols[1:]]
y_pred = model.predict(x_test)

submission = pd.DataFrame({'id': test_df['id'], 'target': y_pred[:, 1]})
submission.to_csv('submission_{}.csv'.format(BUILD_COUNT), index=False)