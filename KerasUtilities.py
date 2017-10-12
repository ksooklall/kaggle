import matplotlib.pyplot as plt

def plot_model(model):
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