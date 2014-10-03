"""
Experiments on sentiment analysis of kaggle movie reviews
"""

import argparse
import datetime
import pickle
import os
import re
import numpy as np
import pandas as pd
import word2vec
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def get_words_from_phrase(phrase, lowercase=True):
    s = phrase.lower() if lowercase else phrase
    return re.findall(r"[a-z']+", s)

class Experiment():
    start_time = datetime.datetime.now()

    def __init__(self, **args):
        self.__dict__ = args
        if not self.config.quiet:
            print("parameters:")
            print(self.config)


    def evaluate_tfidf(self):
# 1st version of vectorization
#unigram_bool = CountVectorizer(encoding='latin-1', min_df=5, stop_words=None, binary=True)
#bigram_count = CountVectorizer(encoding='latin-1', min_df=5, ngram_range=(1,2), stop_words=None)
#unigram_norm = TfidfVectorizer(encoding='latin-1', min_df=5, use_idf=False, smooth_idf=False, norm=u'l1')
#unigram_tfidf = TfidfVectorizer(encoding='latin-1', min_df=5, use_idf=True, smooth_idf=True, norm=u'l2')
        pass

    def evaluate_word2vec(self):
        self._read_training_datafile(datafile="train.tsv")
        self._load_word2vec_model(pretrained_word2vec_model="text8.bin")
        self._convert_phrases_to_vectors()
        self._classify()

    def _read_training_datafile(self, datafile):
        filepath = os.path.join(self.config.data_folder, datafile)
        fields = ('PhraseId', 'SentenceId', 'Phrase', 'Sentiment')
        train = pd.read_csv(filepath, sep='\t', header=0, names=fields)
        self._print_time_elapse(task='read training data')

        play_size = self._play_size(len(train))
        print("dataset size of your experiment: %s" % play_size)

        self.phrases = train['Phrase'].values[:play_size]
        self.y = train['Sentiment'].values[:play_size]

    def _load_word2vec_model(self, pretrained_word2vec_model):
        word2vec_model_filepath = os.path.join(self.config.data_folder, pretrained_word2vec_model)
        self.model = word2vec.load(word2vec_model_filepath)
        self._print_time_elapse('loading word2vec pretrained model')

    def _convert_phrases_to_vectors(self):
        vec_size = len(self.model['the']) # "the" a common word - we use it to obtain the size of the word vector
        zero_row = [0] * vec_size
        self.X = [zero_row] * len(self.phrases)

        model_vocabulary_set = set(self.model.vocab)

        word_vector_hash = {}
        # model[word] is a very expensive operation
        # with hashing, the time used for creating self.X is now reduced from 2000 seconds to about 40 seconds

        phrases_without_word_in_word2vec_vocab = 0
        outside_words = []

        for i, phrase in enumerate(self.phrases):
            if (i+1) % 10000 == 0 and self.debug:
                self._print_time_elapse('%s'%i)

            vec_array = []
            for word in get_words_from_phrase(phrase, lowercase=True):
                if word in word_vector_hash:
                    vec = word_vector_hash[word]
                elif word in model_vocabulary_set:
                    vec = np.array(self.model[word])
                    word_vector_hash[word] = vec
                else:
                    outside_words.append(word)
                    continue
                vec_array.append(vec)

            if len(vec_array)==0:
                phrases_without_word_in_word2vec_vocab += 1
                self.X[i] = np.asarray(zero_row)
            else:
                self.X[i] = self._pooling()(np.asarray(vec_array), axis=0)


        print "number of word2vec word used in movie reviews: ", len(word_vector_hash.keys())
        print "phrases without any word in word2vec_vocab: ", phrases_without_word_in_word2vec_vocab

        self._print_time_elapse('creating word2vec matrix')
    
    def _classify(self):
        #clf = LinearSVC(C=0.0001, penalty='l1', dual=False)
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
    
        if self.config.train_ratio is not None:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=self.config.train_ratio, random_state=0)
            clf.fit(X_train, y_train)
            self._print_time_elapse('finishing training')

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print 'accuracy: ', acc
            self._print_time_elapse('finishing prediction on test set (test size=%s)' % len(X_test))

        else:
            scores = cross_validation.cross_val_score(clf, self.X, self.y, cv=self.config.cv, verbose=2, n_jobs=-1)
            print 'scores: ', scores
            self._print_time_elapse('finishing CV validation')

    def _pooling(self):
        if self.config.pooling == 'max':
            return np.amax
        elif self.config.pooling == 'min':
            return np.amin
        elif self.config.pooling == 'mean':
            return np.mean
        elif self.config.pooling == 'sum':
            return np.sum
        else:
            raise Exception("Wrong pooling parameter - must be one of max, min, mean, sum")

    def _play_size(self, fullsize):
        size = int(fullsize * self.config.play_ratio)
        if size > fullsize:
            raise Exception('play_ratio should be smaller than 1.0')
        elif size < 100:
            raise Exception('play_ratio is too small, and as a result, play_size is smaller than 100')
        else: 
            return size

    def _print_time_elapse(self, task):
        print("time elapsed after {task}: {time}".format(task=task, time=datetime.datetime.now() - Experiment.start_time) )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv', default=2, type=int, help='cross-validation folder number (default=2)')
    parser.add_argument('--play_ratio', default=1.0, type=float, help='how much of training data will be used (default=1.0). Range: (0.,1.]')
    parser.add_argument('--data_folder', default='../data', help='folder of word2vec model/corpus, train/test data (default="../data")')
    parser.add_argument('--train_ratio', type=float, help='training ratio. For example, 0.3, 0.5, 0.7')
    parser.add_argument('--quiet', default=False, action="store_true", help='don\'t show verbose info')
    parser.add_argument('--pooling', default="max", help='how to aggregate a set of word vectors into one vector (default="max") // choices: max, min, sum, mean')

    config = parser.parse_args()

    experiment = Experiment(config=config)
    experiment.evaluate_word2vec()

if __name__ == "__main__":
    main()
