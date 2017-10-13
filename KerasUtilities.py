from keras import initializers
from keras import regularizers
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential, Model, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt


def vanilla_nn(X_train, y_train, hidden_units, drop_out, batch_size, depth):
    """
    A feed_foward network
    S_order(2)
    :param hidden_units:
    :param drop_out:
    :param batch_size:
    :param depth:
    :return:
    """
    model = Sequential()
    regularizers.l2(0.01)
    trunc_norm = initializers.truncated_normal(stddev=0.01)
    model.add(Dense(hidden_units, kernel_initializer=trunc_norm, activation='relu', input_shape=(X_train.shape[1],)))

    for i in range(depth):
        model.add(Dense(hidden_units//2, kernel_initializer=trunc_norm, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(drop_out))

    model.add(Dense(hidden_units//2, kernel_initializer=trunc_norm, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    model.summary()

    checkpointer = ModelCheckpoint(filepath='model.weights.best.h5', verbose=1, save_best_only=True)
    model_info = model.fit(X_train, y_train, batch_size=batch_size, epochs=100, validation_split=0.2, callbacks=[checkpointer], verbose=1)

    return model, model_info


def plot_model(model):
    """
    Plotting graph
    :param model: Keras model
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].plot(range(1, len(model.history['acc']) + 1), model.history['acc'])
    ax[0].plot(range(1, len(model.history['val_acc']) + 1), model.history['val_acc'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(range(1, len(model.history['loss']) + 1), model.history['loss'])
    ax[1].plot(range(1, len(model.history['val_loss']) + 1), model.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')