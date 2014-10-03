nlp-movie-review
================

Usage:

1. Create a data folder, and put your data file "text8.bin", "train.tsv" in the folder.

2. Examples:

> python experiment.py -h

```
usage: experiment.py [-h] [--cv CV] [--play_ratio PLAY_RATIO]
                     [--data_folder DATA_FOLDER] [--train_ratio TRAIN_RATIO]
                     [--quiet] [--pooling POOLING]

Experiments on sentiment analysis of kaggle movie reviews

optional arguments:
  -h, --help            show this help message and exit
  --cv CV               cross-validation folder number (default=2)
  --play_ratio PLAY_RATIO
                        how much of training data will be used (default=1.0).
                        Range: (0.,1.]
  --data_folder DATA_FOLDER
                        folder of word2vec model/corpus, train/test data
                        (default="../data")
  --train_ratio TRAIN_RATIO
                        training ratio. For example, 0.3, 0.5, 0.7
  --quiet               don't show verbose info
  --pooling POOLING     how to aggregate a set of word vectors into one vector
                        (default="max") // choices: max, min, sum, mean
```

> python experiment.py --play_ratio 0.01 --cv 3

```
use only 1% data (i.e., 1560 phrases out of 156,000) for experiment
```

> python experiment.py --cv 3 --pooling min

```
use min-pooling to aggregate vectors of all words in a phrase
```

> python experiment.py --data_folder ../data --play_ratio 0.01

```
parameters:
    data_folder = ../data
    play_ratio = 0.01
    quiet = False
    train_ratio = None
    pooling = max
    cv = 2
time elapsed after read training data: 0:00:00.229506
dataset size of your experiment: 1560
time elapsed after loading word2vec pretrained model: 0:00:04.652768
number of word2vec word used in movie reviews:  504
phrases without any word in word2vec_vocab:  59
time elapsed after creating word2vec matrix: 0:00:05.458077
score: 0.671795
[Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    1.1s
score: 0.669231
[Parallel(n_jobs=-1)]: Done   2 out of   2 | elapsed:    1.1s finished
scores:  [ 0.67179487  0.66923077]
time elapsed after finishing CV validation: 0:00:06.721579
```
