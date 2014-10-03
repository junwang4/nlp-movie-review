nlp-movie-review
================

Usage:

1. Create a data folder, and put your data file "text8.bin", "train.tsv" in the folder.

2. Examples:

> python experiment.py -h

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


> python experiment.py --play_ratio=0.01 --cv 3

> python experiment.py --cv 3 --pooling min
