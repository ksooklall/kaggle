from keras import initializers
from keras import regularizers
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential, Model, Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout

import matplotlib.pyplot as plt


def vanilla_nn(X_train, y_train, hidden_units, drop_out, batch_size, depth, loss, filepath):
    """
    A feed_foward network
    S_order(2)
    :param hidden_units:
    :param drop_out:
    :param batch_size:
    :param depth:
    :return:
    """
    if not isinstance(hidden_units, int):
        raise ValueError('Hidden units must be an int, found {}'.format(hidden_units))

    if not isinstance(drop_out, float):
        raise ValueError('Hidden units must be an int, found {}'.format(drop_out))

    if not isinstance(batch_size, int):
        raise ValueError('Hidden units must be an int, found {}'.format(batch_size))

    if not isinstance(depth, int):
        raise ValueError('Hidden units must be an int, found {}'.format(depth))

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
    model.add(Dense(y_train.shape[-1], activation='softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    model.summary()

    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    model_dict = model.fit(X_train, y_train, batch_size=batch_size, epochs=25, validation_split=0.2,
                           callbacks=[checkpointer], verbose=1).history
    plot_model(model_dict)
    return model, model_dict


def plot_model(model_dict):
    """
    Plotting graph
    :param model: Keras model
    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].plot(range(1, len(model_dict['acc']) + 1), model_dict['acc'])
    ax[0].plot(range(1, len(model_dict['val_acc']) + 1), model_dict['val_acc'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].plot(range(1, len(model_dict['loss']) + 1), model_dict['loss'])
    ax[1].plot(range(1, len(model_dict['val_loss']) + 1), model_dict['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    plt.show()