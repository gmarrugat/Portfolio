# RNN approach for NER

import numpy as np
from keras.layers import Input, Embedding, Dense, Bidirectional, GRU, TimeDistributed, Dropout, concatenate
from keras.models import Model

from data import Data
from util import print_time, hasdigits, is_short_word, is_medium_short_word, is_medium_long_word, is_long_word
from results import metrics_to_report, plot_loss_acc


def build_model(data, n_features):
    inp = Input(shape=(data.max_len,))
    feat = Input(shape=(data.max_len, n_features))
    x = Embedding(input_dim=data.n_words, output_dim=50, input_length=data.max_len)(inp)
    x = concatenate([x, feat], axis=-1)
    x = Dropout(0.1)(x)
    # model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    x = Bidirectional(GRU(units=100, return_sequences=True, recurrent_dropout=0.1))(x)
    out = TimeDistributed(Dense(data.n_tags, activation="softmax"))(x)

    m = Model(inputs=[inp, feat], outputs=out)
    m.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return m


def get_features(data, X):
    features = []
    for sent in X:
        sent_features = []
        for pos, w_idx in enumerate(sent):
            word = data.pos_to_word[w_idx]
            word_features = []
            word_features.append(1 if word.isdigit() else 0)
            word_features.append(1 if hasdigits(word) else 0)
            word_features.append(1 if word.istitle() else 0)
            word_features.append(1 if word.isupper() else 0)
            word_features.append(1 if word.find("-") != -1 else 0)
            word_features.append(1 if word.find("'") != -1 else 0)
            word_features.append(1 if word.find(".") != -1 else 0)
            word_features.append(1 if is_short_word(word) else 0)
            word_features.append(1 if is_medium_short_word(word) else 0)
            word_features.append(1 if is_medium_long_word(word) else 0)
            word_features.append(1 if is_long_word(word) else 0)
            sent_features.append(word_features)

        features.append(sent_features)
    return np.array(features)


def tag_ohe_to_id(data, X, Y):
    Y_tags = []
    Y_tags_flatten = []

    for sentence, tags in zip(X, Y):
        sentence_tags = []

        for word, tag in zip(sentence, tags):
            if word == 0:  # reached EOC
                break
            tag = np.argmax(tag)
            sentence_tags.append(tag)
            Y_tags_flatten.append(tag)

        Y_tags.append(sentence_tags)

    return Y_tags, Y_tags_flatten


def main(params):
    # load data
    data = Data(params["data_file"])
    data.fetch(n_sentences=params["number_sentences"], padding=True)

    # generate features
    train_features = get_features(data, data.X_train)
    test_features = get_features(data, data.X_test)
    n_features = train_features.shape[2]
    print("features shape (train):", train_features.shape)
    print("features shape (test):", test_features.shape)

    # build RNN model
    model = build_model(data, n_features)
    model.summary()

    # train RNN model
    history = model.fit(
        [data.X_train, train_features],
        data.y_train,
        batch_size=32,
        epochs=params["number_epochs"],
        validation_data=([data.X_test, test_features], data.y_test))
    print()

    # model convergence
    print('Model convergence...')
    plot_loss_acc(history, filename='Loss_Acc' + params['output_sufix'])
    print()

    # METRICS
    list_classes = np.array(list(data.pos_to_tag.values()))

    print('Metrics for TRAIN SET:')
    y_train_pred = model.predict([data.X_train, train_features]) # predict on train set
    y_train_true_tags, y_train_true_tags_flatten = tag_ohe_to_id(data, data.X_train, data.y_train)
    y_train_pred_tags, y_train_pred_tags_flatten = tag_ohe_to_id(data, data.X_train, y_train_pred)
    metrics_to_report(
        y_train_true_tags_flatten, y_train_pred_tags_flatten,
        y_train_true_tags, y_train_pred_tags,
        list_classes, params['output_sufix']
    )
    print()

    print('Metrics for TEST SET:')
    y_test_pred = model.predict([data.X_test, test_features]) # predict on test set
    y_test_true_tags, y_test_true_tags_flatten = tag_ohe_to_id(data, data.X_test, data.y_test)
    y_test_pred_tags, y_test_pred_tags_flatten = tag_ohe_to_id(data, data.X_test, y_test_pred)
    metrics_to_report(
        y_test_true_tags_flatten, y_test_pred_tags_flatten,
        y_test_true_tags, y_test_pred_tags,
        list_classes, params['output_sufix']
    )
    print()

    return


if __name__ == '__main__':
    params = {
        'data_file': '../data/ner_dataset.csv',
        'number_sentences': 100,
        'number_epochs': 10,
        'output_sufix': '_RNN',
    }
    main(params)
