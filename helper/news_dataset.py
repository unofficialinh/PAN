import numpy as np
import torchvision
import helper.pu_learning_dataset as pu_learning_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
__author__ = 'garrett_local'


def _prepare_20news_data():
    train_x, train_y = fetch_20newsgroups(subset="train", return_X_y=True, data_home='./Dataset/20News')
    test_x, test_y = fetch_20newsgroups(subset="test", return_X_y=True, data_home='./Dataset/20News')

    # pdb.set_trace()
    # Binarize labels.
    train_y[train_y > 10] = 19
    train_y[train_y <= 10] = 1
    train_y[train_y == 19] = 0
    test_y[test_y > 10] = 19
    test_y[test_y <= 10] = 1
    test_y[test_y == 19] = 0

    tf = TfidfVectorizer(stop_words='english', max_features=784)
    train_x = tf.fit_transform(train_x)
    train_x = train_x.toarray()
    test_x = tf.transform(test_x)
    test_x = test_x.toarray()
    return train_x, train_y, test_x, test_y


class NewsDataset(pu_learning_dataset.PuLearningDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_20news_data()
        super(NewsDataset, self).__init__(*args, **kwargs)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y


class NewsPnDataset(pu_learning_dataset.PnLearningDataSet):

    def __init__(self, *args, **kwargs):
        self._train_x, self._train_y, self._test_x, self._test_y = \
            _prepare_20news_data()
        super(NewsPnDataset, self).__init__(*args, **kwargs)

    def _original_train_x(self):
        return self._train_x

    def _original_train_y(self):
        return self._train_y

    def _original_test_x(self):
        return self._test_x

    def _original_test_y(self):
        return self._test_y