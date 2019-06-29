
# SP approach for NER

import os, sys, inspect
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

import skseq
from skseq.sequences import sequence
from skseq.sequences.sequence import Sequence
from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary
from skseq.sequences import extended_feature

import skseq.readers.pos_corpus
import skseq.sequences.structured_perceptron as spc

from data import Data
from util import print_time, hasdigits, is_short_word, is_medium_short_word, is_medium_long_word, is_long_word, compare_true_predict
from results import metrics_to_report, plot_loss_acc, plot_acc


def flatten(Y):
    return [tag for sentence in Y for tag in sentence]


def tags_to_ids(data, Y):
    return [[data.tag_to_pos[tag] for tag in sentence] for sentence in Y]


def ids_to_tags(data, Y):
    return [[data.pos_to_tag[tag] for tag in sentence] for sentence in Y]


def main(params):

    # load data
    N_SENT = params["number_sentences"]
    data = Data(params["data_file"])
    data.fetch(n_sentences=N_SENT)

    train_size = len(data.X_train)
    test_size = len(data.X_test)
    print("train_size:", train_size, " test_size:", test_size, " total:", train_size+test_size)
    if N_SENT is None:
        N_SENT = 47959
    assert N_SENT == train_size + test_size

    # generate sequences
    sequence_list = SequenceList(LabelDictionary(data.word_to_pos), LabelDictionary(data.tag_to_pos))
    for i in tqdm(range(train_size)):
        x,y = data.X_train[i], data.y_train[i]
        sequence_list.add_sequence(x,y, LabelDictionary(data.word_to_pos), LabelDictionary(data.tag_to_pos))

    # generate features
    ex_feature_mapper = skseq.sequences.extended_feature.ExtendedFeatures(sequence_list)
    ex_feature_mapper.build_features()
    feature_mapper = ex_feature_mapper
    #print("Number of features:", len(feature_mapper.feature_dict), len(feature_mapper.feature_list))
    features = set([x.split(":")[0] for x in feature_mapper.feature_dict.keys()])
    print("Features:", features)

    # build SP model
    corpus = skseq.readers.pos_corpus.PostagCorpus()
    sp = spc.StructuredPerceptron(data.word_to_pos, data.tag_to_pos, feature_mapper)

    # train SP model
    num_epochs = params["number_epochs"]
    sp.fit(feature_mapper.dataset, num_epochs, tolerance=0.0005)
    print()

    # model convergence
    print('Model convergence...')
    plot_acc(sp.acc_per_epoch, filename='Acc'+params['output_sufix'])
    print()

    # METRICS
    list_classes = np.array(list(data.pos_to_tag.values()))

    print('Metrics for TRAIN SET:')
    y_train_true_tags = tags_to_ids(data, data.y_train)
    y_train_true_tags_flatten = flatten(y_train_true_tags)
    y_train_pred_tags = [list(sp.predict_tags_given_sentence(sentence)[0].y) for sentence in data.X_train] # predict on train set
    y_train_pred_tags_flatten = flatten(y_train_pred_tags)
    wrong_sentences_ids_train = metrics_to_report(
        y_train_true_tags_flatten, y_train_pred_tags_flatten,
        y_train_true_tags, y_train_pred_tags,
        list_classes, params['output_sufix']
    )
    print()

    print('Metrics for TEST SET:')
    y_test_true_tags = tags_to_ids(data, data.y_test)
    y_test_true_tags_flatten = flatten(y_test_true_tags)
    y_test_pred_tags = [list(sp.predict_tags_given_sentence(sentence)[0].y) for sentence in data.X_test] # predict on test set
    y_test_pred_tags_flatten = flatten(y_test_pred_tags)
    wrong_sentences_ids_test = metrics_to_report(
        y_test_true_tags_flatten, y_test_pred_tags_flatten,
        y_test_true_tags, y_test_pred_tags,
        list_classes, params['output_sufix']
    )
    print()

    if params["max_wrong_samples"] is not None:
        print("Some wrong predictions in the test set:\n")
        for id in wrong_sentences_ids_test[:params["max_wrong_samples"]]:
            sentence = data.X_test[id]
            true = data.y_test[id]
            pred = [data.pos_to_tag[tag] for tag in y_test_pred_tags[id]]
            compare_true_predict(sentence, true, pred)
        print()

    return


if __name__ == '__main__':
    params = {
        'data_file': '../data/ner_dataset.csv',
        'number_sentences': 100,
        'number_epochs': 10,
        'output_sufix': '_SP',
        'max_wrong_samples': 10,
    }
    main(params)
