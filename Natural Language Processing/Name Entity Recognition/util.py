import pickle
import datetime
import os
import numpy as np

def save_instance(instance, path='../data/data.pkl'):
    with open(path, 'wb') as output:
        pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)


def load_instance(path='../data/data.pkl'):
    with open(path, 'rb') as inp:
        return pickle.load(inp)


def print_time(string=''):
    print(str(datetime.datetime.today())[:22], string)
    return


def create_directory(path):
    try:
        os.mkdir(path)
        print("Directory", path, "was created!")
    except:
        print("Directory", path, "already exists!")


def hasdigits(word):
    return any(char.isdigit() for char in word)


def is_short_word(word):
	if len(word) < 4:
		return True
	else:
		return False

def is_medium_short_word(word):
	if len(word) > 3 and len(word) < 6:
		return True
	else:
		return False

def is_medium_long_word(word):
	if len(word) > 5 and len(word) < 8:
		return True
	else:
		return False

def is_long_word(word):
	if len(word) > 7:
		return True
	else:
		return False


def compare_true_predict(sentence, true, pred):
    widths = [15,6,6]
    print("WORD:".ljust(widths[0]) + "TRUE:".ljust(widths[1]) + "PRED:".ljust(widths[2]))
    print("-"*np.sum(widths))
    for word, tag_true, tag_pred in zip(sentence, true, pred):
        print(word.ljust(widths[0]) + tag_true.ljust(widths[1]) + tag_pred.ljust(widths[2]))
    print()


def notify(title, subtitle, message):
    t = '-title {!r}'.format(title)
    s = '-subtitle {!r}'.format(subtitle)
    m = '-message {!r}'.format(message)
    os.system('terminal-notifier {}'.format(' '.join([m, t, s])))
