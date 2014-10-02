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


START_TIME = datetime.datetime.now()

DATA_FOLDER = "../data"
FIELDS = ('PhraseId', 'SentenceId', 'Phrase', 'Sentiment')

def print_time_elapse(task):
    print("time elapsed after {task}: {time}".format(task=task, time=datetime.datetime.now() - START_TIME) )

def read_training_datafile(datafile="train.tsv"):
    filepath = os.path.join(DATA_FOLDER, datafile)
    train = pd.read_csv(filepath, sep='\t', header=0, names=FIELDS)
    print_time_elapse(task='read training data')
    X=train['Phrase'].values
    y=train['Sentiment'].values
    return X, y

def experiment_with_tfidf():
# 1st version of vectorization
#unigram_bool = CountVectorizer(encoding='latin-1', min_df=5, stop_words=None, binary=True)
#unigram_count = CountVectorizer(encoding='latin-1', min_df=5, stop_words=None)
#unigram_norm = TfidfVectorizer(encoding='latin-1', min_df=5, use_idf=False, smooth_idf=False, norm=u'l1')
#unigram_tfidf = TfidfVectorizer(encoding='latin-1', min_df=5, use_idf=True, smooth_idf=True, norm=u'l2')
#bigram_count = CountVectorizer(encoding='latin-1', min_df=5, ngram_range=(1,2), stop_words=None)
    pass

def experiment_with_word2vec():
    """vector dimension of text8.bin: 300
    
    What is the corpus used for generating text8.bin ?

    Note:
        We should not use "neighbor words" - look at this example:
            > model.cosine('good')
            > 
            >[('bad', 0.54042731028657409),
            > ('natured', 0.46385842768073826),
            > ('luck', 0.42497662752295273),
            > ('egun', 0.4212025679591751),
            > ('affectation', 0.39169068231063342),
            > ('pretty', 0.37954450816919627),
            > ('foolish', 0.37181316813704901),
            > ('reasonable', 0.36360311062209416),
            > ('goodness', 0.35954953294022562),
            > ('you', 0.3550184667540342)]
    """

    X, y = read_training_datafile(datafile="train.tsv")

    word2vec_vec = load_word2vec_array_from_cache() # we would save some time when loading the preprocessed word2vec data from a pickled file.
    # the cached file is 1.11G, takes almost 1 minute to dump, and takes 23 seconds to load

    if word2vec_vec is None:
        word2vec_model_file = os.path.join(DATA_FOLDER, "text8.bin")
        model = word2vec.load(word2vec_model_file)
        print_time_elapse('loading word2vec pretrained model')

        vec_dimensions = len(model['the']) # all models should have "the" in their vocabulary, so we use it to obtain the word vector dimension
        zero_row = [0] * vec_dimensions
        word2vec_vec = [zero_row] * len(X)

        model_vocabulary_set = set(model.vocab)
        # valid words in word2vec (text8.bin): 12,226
        # cnt of phrases whose all words are outside word2vec (71,291): 3430

        # model[word] is a very expensive operation, so we need to hash it
        # with hashing, the time used for creating the matrix is now reduced from 2000 seconds to about 50 seconds
        word_vector_hash = {}

        words_outside_word2vec = 0
        for i in range(0, len(X)):
            words_list = re.findall(r"[a-z']+", X[i].lower())

            if (i+1)% 10000 == 0: print_time_elapse('%s'%i)

            data = []
            for word in words_list:
                if word in word_vector_hash:
                    v = word_vector_hash[word]
                elif word in model_vocabulary_set:
                    v = np.array(model[word])
                    word_vector_hash[word] = v
                else:
                    continue
                data.append(v)

            if len(data)==0:
                #print words_list
                words_outside_word2vec += 1
                word2vec_vec[i] = np.asarray(zero_row)
            else:
                #word2vec_vec[i] = np.sum(np.asarray(data), axis=0)
                #word2vec_vec[i] = np.mean(np.asarray(data), axis=0)
                word2vec_vec[i] = np.amax(np.asarray(data), axis=0)

        print "word cnt:", len(word_vector_hash.keys())
        print "outside word cnt:", words_outside_word2vec
        dump_word2vec_array(word2vec_vec)

    print_time_elapse('creating word2vec matrix')

    #clf = LinearSVC(C=0.0001, penalty='l1', dual=False)
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)

    use_cross_validation = True
    if use_cross_validation:
        CV_FOLD = 3
        scores = cross_validation.cross_val_score(clf, word2vec_vec, y.astype(int), cv=CV_FOLD)
        print 'scores: ', scores
        print_time_elapse('finishing CV validation')
    else:
        X_train, X_test, y_train, y_test = train_test_split(word2vec_vec, y, train_size=0.5, random_state=0)
        clf.fit(X_train, y_train)
        print_time_elapse('finishing training')

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print 'accuracy: ', acc
        print_time_elapse('finishing prediction on test set (test size=%s)'%len(X_test))


def word2vec_pickle_file():
    #return os.path.join(DATA_FOLDER, "train_word2vec.pickle")
    return os.path.join("/tmp", "train_word2vec.pickle")

def dump_word2vec_array(word2vec_vec):
    filepath = word2vec_pickle_file()
    if not os.path.exists(filepath):
        pickle.dump(word2vec_vec, open(filepath, "wb" ) )

def load_word2vec_array_from_cache():
    filepath = word2vec_pickle_file()
    if os.path.exists(filepath):
        return pickle.load(open(filepath, "rb" ) )
    else:
        return None 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv', default=2, type=int, help='cross-validation folder number (default=2)')
    config = parser.parse_args() # TODO (Jun Wang)

    experiment_with_word2vec()

if __name__ == "__main__":
    main()
