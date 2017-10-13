import tqdm
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


#from helper import clean_columns, standerize, up_sampling, down_sampling
#from KerasUtilities import plot_model

from keras import initializers
from keras import regularizers
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential, Model, Input, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout

BUILD_COUNT = 8
SEED = BUILD_COUNT ** 2

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df[train_df == -1] = np.nan
#train_df = clean_columns(train_df)
import pdb;

# Nan columns kept and replaced
nan_train_df = train_df.fillna(-999)
# Nan columns removed
full_cols = train_df.columns[~(train_df.isnull().sum() > 0)]

# Skewed cols, check ps_calc_06
skewed_cols = ["ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin","ps_ind_13_bin", "ps_car_11_cat"]
train_cols = list(set(full_cols).difference(skewed_cols))

full_train_df = train_df[train_cols]
#mean, std, std_train_df = standerize(full_train_df.drop(['target'], axis=1))


def get_training_testing(df, one_hot=True):
    if 'target' not in df.columns:
        df['target'] = train_df['target']

    X_train = np.array(df.drop(['target', 'id'], axis=1)).astype(np.float32)
    y_train = np.array(df['target'])
    if one_hot:
        y_train = np_utils.to_categorical(y_train, 2)

    return X_train, y_train

#re_sampling_df = down_sampling(full_train_df, SEED)
X_df = full_train_df[full_train_df['target'] == 0]

X, y, = get_training_testing(full_train_df, one_hot=False)
X_train, X_test = train_test_split(full_train_df, test_size=0.2, random_state=SEED)
X_train = X_train[X_train['target'] == 0]
X_train = X_train.drop(['target', 'id'], axis=1)

y_test = X_test['target']
X_test = X_test.drop(['target', 'id'], axis=1)

X_train = X_train.values
X_test = X_test.values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#X_train, X_valid, y_train, y_valid = X_train_valid[:split], X_train_valid[split:], y_train_valid[:split], y_train_valid[split:]
#X_valid, X_test, y_valid, y_test = X_valid * std + mean, X_test * std + mean, y_valid * std + mean, y_test * std + mean


# Hyperparameters
hidden_units = 128
drop_out = 0.3
batch_size = 512


def auto_encoder(X, y, encoded_dim):
    input_dim=X.shape[1]
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoded_dim, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(encoded_dim//2, activation='relu')(encoder)
    decoder = Dense(encoded_dim//4, activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    autoencoder.summary()
    time.sleep(5)
    return train_and_save(autoencoder, X, y)


def train_and_save(model, X, y):
    checkpointer = ModelCheckpoint(filepath='auto_encoder.weights.h5', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_history = model.fit(X, y, batch_size=batch_size, epochs=25,
                        validation_split=0.2, callbacks=[checkpointer, tensorboard], verbose=1).history
    return model_history


def predict_and_submit(model):
    x_test = np.array(test_df[[i for i in train_cols if i != 'target']])
    y_pred = model.predict(x_test)

model = auto_encoder(X_train, X_train, hidden_units)
import pdb; pdb.set_trace()
