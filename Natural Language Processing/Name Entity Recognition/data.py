import numpy as np
import pandas as pd
#from tqdm import tqdm_notebook as tqdm
from util import print_time

from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, filename):
        self.filename = filename
        self.max_len, self.n_words, self.n_tags, self.X_train, self.X_test, self.y_train, self.y_test, self.df = [None] * 8
        self.word_to_pos, self.pos_to_word , self.tag_to_pos, self.pos_to_tag = [None] * 4

    # Words in dataset
    @staticmethod
    def build_word_to_pos(X,padding="EOC"):
        word_to_pos = {}
        word_to_pos[padding] = 0
        i = 1
        print_time("building build_word_to_pos...")
        for s in X:
            for w in s:
                if w not in word_to_pos:
                    word_to_pos[w] = i
                    i += 1

        pos_to_word = {v: k for k, v in word_to_pos.items()}
        return word_to_pos, pos_to_word

    # Labels of each word in dataset
    @staticmethod
    def build_tag_to_pos(Y):
        tag_to_pos = {}
        i = 0
        print_time("building build_tag_to_pos...")
        for s in Y:
            for t in s:
                if t not in tag_to_pos:
                    tag_to_pos[t] = i
                    i += 1
        pos_to_tag = {v: k for k, v in tag_to_pos.items()}

        return tag_to_pos, pos_to_tag

    def fetch(self, n_sentences=None, test_perc=0.25, padding=False):
        X = []
        Y = []

        self.df = pd.read_csv(self.filename, encoding="unicode_escape")

        sentence_formatter = "Sentence: {}"

        # Define the range of sentences to consider
        # and number them in the data frame
        if n_sentences is not None:
            print_time("loading " + str(n_sentences) + " sentencens...")
            end = self.df.index[self.df["Sentence #"] == sentence_formatter.format(n_sentences+1)][0]
            self.df = self.df[0:end]
            #last_n = n_sentences - 1
        else:
            print_time("loading full dataset...")
            #n_sentences = len(list(set(self.df['Sentence #'])))
            n_sentences = len(list(set(self.df["Sentence #"]) - set([np.nan])))
            #last_n = n_sentences - 1
        first_n = 1
        last_n = n_sentences
        n_rows = self.df.shape[0]

        print_time("extending sentence identifier and building X and Y...")
        self.df.fillna(value="",inplace=True)
        current_sent_id = sentence_formatter.format(-1)
        sentence_words = []
        sentence_tags = []
        for index, row in self.df.iterrows():
            print("  row {}/{}".format(index,n_rows), end="\r")
            if row["Sentence #"] != "": # new sentence
                X.append(sentence_words)
                Y.append(sentence_tags)
                sentence_words = []
                sentence_tags = []
                current_sent_id = row["Sentence #"]
            row["Sentence #"] = current_sent_id
            sentence_words.append(row["Word"])
            sentence_tags.append(row["Tag"])
        print()
        X.append(sentence_words)
        Y.append(sentence_tags)
        if len(X[0]) == 0: # delete first empty sentence
            del X[0]
            del Y[0]

        #print_time("extending sentence identifier...")
        #for s_id in tqdm(range(first_n, last_n)):
        #    sentence_id = sentence_formatter.format(s_id)
        #    sentence_id_next = sentence_formatter.format(s_id + 1)
        #    start = self.df.index[self.df["Sentence #"] == sentence_id][0]
        #    end = self.df.index[self.df["Sentence #"] == sentence_id_next][0]
        #    self.df["Sentence #"][start:end] = sentence_id
        #sentence_id = sentence_formatter.format(last_n)
        #start = self.df.index[self.df["Sentence #"] == sentence_id][0]
        #end = self.df.shape[0]
        #self.df["Sentence #"][start:end] = sentence_id

        # Build X and Y
        #print_time("building X and Y...")
        #for i in range(first_n, last_n+1):
        #    #print(" sentence {}/{}".format(i,last_n), end="\r")
        #    s = sentence_formatter.format(i)
        #    X.append(list(self.df[self.df["Sentence #"] == s]["Word"].values))
        #    Y.append(list(self.df[self.df["Sentence #"] == s]["Tag"].values))
        #print()

        self.word_to_pos, self.pos_to_word = self.build_word_to_pos(X)
        self.tag_to_pos, self.pos_to_tag = self.build_tag_to_pos(Y)

        self.max_len = max([len(x) for x in X])
        self.n_words = len(self.word_to_pos)
        self.n_tags = len(self.tag_to_pos)

        if padding:
            from keras.utils import to_categorical
            from keras.preprocessing.sequence import pad_sequences

            X = [[self.word_to_pos[w] for w in s] for s in X]  # convert the dataset into the index of dictionary word_to_pos
            Y = [[self.tag_to_pos[t] for t in s] for s in Y]  # convert the labels into the index of dictionary tag_to_pos

            X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=0)
            Y = pad_sequences(maxlen=self.max_len, sequences=Y, padding="post", value=self.tag_to_pos["O"])
            Y = np.array([to_categorical(i, num_classes=self.n_tags) for i in Y])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=test_perc, shuffle=False)
