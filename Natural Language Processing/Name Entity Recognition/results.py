import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(y_true, y_pred, classes, filename='Confusion_Matrix', normalize=False, title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix saved")
        pass
    else:
        #print('Confusion matrix, without normalization saved')
        pass

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    file_path = '../results/' + filename + ".png"
    fig.savefig(file_path, bbox_inches='tight')
    print("Confusion Matrix saved on", file_path)

    return ax


def re(y_true, y_pred):
    correct_pred_sentences = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct_pred_sentences += 1

    perc_correct_pred_sentences = correct_pred_sentences / len(y_pred)

    print('Sentences without any error:', correct_pred_sentences, ', {:.2%}'.format(perc_correct_pred_sentences),
          'of the total sentences')


def plot_loss_acc(history, filename='Loss_Acc'):
    fig = plt.figure()
    fig.set_size_inches(15, 7)

    fig.add_subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='train acc')
    plt.plot(history.history['val_acc'], label='val acc')
    plt.xlabel('epochs')
    plt.title('Accuracy')

    plt.legend()

    fig.add_subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('epochs')
    plt.title('Loss')

    plt.legend()

    fig.suptitle('Training Loss and Accuracy', fontsize=16)

    file_path = '../results/' + filename + ".png"
    fig.savefig(file_path, bbox_inches='tight')
    print("Loss-Accuracy plot saved on", file_path)


def sentences_without_any_error(y_true, y_pred):
    correct_pred_sentences = 0
    wrong_sentences_ids = []

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct_pred_sentences += 1
        else:
            wrong_sentences_ids.append(i)
        #print(type(y_pred[i]), y_pred[i])
        #print(type(y_true[i]), y_true[i])

    perc_correct_pred_sentences = correct_pred_sentences / len(y_pred)
    print('Sentences without any error:', correct_pred_sentences, ', {:.2%}'.format(perc_correct_pred_sentences),
          'of the total sentences')
    return wrong_sentences_ids


def plot_acc(accuracy, filename='Accuracy'):
    fig = plt.figure()

    plt.plot(accuracy, label="train accuracy")
    plt.title('SP model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()

    file_path = '../results/' + filename + ".png"
    fig.savefig(file_path, bbox_inches='tight')
    print("Accuracy plot saved on", file_path)


def metrics_to_report(y_true_tags, y_pred_tags, y_true_sentences, y_pred_sentences, list_classes, filename):
    #print('Computing Accuracy...')
    accuracy = accuracy_score(y_true_tags, y_pred_tags)
    print('Accuracy:', '{:.4}'.format(accuracy))

    #print('Computing Sentences without any error...')
    wrong_sentences_ids = sentences_without_any_error(y_true_sentences, y_pred_sentences)

    #print('Computing Confusion Matrix...')
    plot_confusion_matrix(y_true_tags, y_pred_tags, list_classes, normalize=True, filename='Confusion_Matrix'+filename)

    return wrong_sentences_ids
