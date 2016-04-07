# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:25:11 2016

@author: rtaromax
"""

import pandas as pd
import numpy as np
from pandas import HDFStore
import pickle
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import bio_f_score
from sklearn.metrics import roc_auc_score

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.externals import six

from seqlearn._utils import atleast2d_or_csr, safe_sparse_dot, validate_lengths

def predict(self, X, lengths=None):
    X = atleast2d_or_csr(X)
    scores = safe_sparse_dot(X, self.coef_.T)
    if hasattr(self, "coef_trans_"):
        n_classes = len(self.classes_)
        coef_t = self.coef_trans_.T.reshape(-1, self.coef_trans_.shape[-1])
        trans_scores = safe_sparse_dot(X, coef_t.T)
        trans_scores = trans_scores.reshape(-1, n_classes, n_classes)
    else:
        trans_scores = None
        
    decode = self._get_decoder()

    if lengths is None:
        y = decode(scores, trans_scores, self.intercept_trans_,
                   self.intercept_init_, self.intercept_final_)
    else:
        start, end = validate_lengths(X.shape[0], lengths)
    
        y = [decode(scores[start[i]:end[i]], trans_scores,
                    self.intercept_trans_, self.intercept_init_,
                    self.intercept_final_)
            for i in six.moves.xrange(len(lengths))]
        y = np.hstack(y)
    
    return self.classes_[y], scores


def 



for i in range(1):

    df_train = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/training_data_'+str(i+27)+'.pickle')
    df_train_label = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/training_labels_'+str(i+27)+'.pickle')
    df_test = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/test_data_'+str(i+27)+'.pickle')
    df_test_label = pd.read_pickle('/Users/rtaromax/Documents/cdc/data_per_week/test_labels_'+str(i+27)+'.pickle')
    
    x_train = df_train.sort_index(level=1)
    y_train = df_train_label.sort_index(level=1)
    
    x_test = df_test.sort_index(level=1)
    y_test = df_test_label.sort_index(level=1)
    
    lengths = list(x_train.count(level=1).ix[:,0])
    lengths_test = list(x_test.count(level=1).ix[:,0].replace(to_replace=0, value=np.nan).dropna())
    
    clf = StructuredPerceptron(lr_exponent=0.001, max_iter=100, random_state=2)
    clf.fit(x_train, y_train, lengths)
    
    
    
    pred, pred_scores = predict(clf, x_test, lengths_test)
    '''
    df_pred_scores = pd.DataFrame(pred_scores, index=df_test_label.index)
    df_pred_scores.columns = ['ScoreCls0','ScoreCls1','ScoreCls2','ScoreCls3','ScoreCls4']
    df_pred_scores.reset_index(inplace=True)
    '''    
    
    pred_label_new = []    
    for pred_label in pred:
        if pred_label < 1:
            pred_label_new.append(1)
        else:
            pred_label_new.append(0)
    
    
    y_true = np.asarray(list(y_test))
    test_label_new = []    
    for test_label in y_true:
        if test_label < 1:
            test_label_new.append(1)
        else:
            test_label_new.append(0)
        
    
    print(roc_auc_score(test_label_new, pred_label_new))
        
    #df_pred_scores.to_csv('/Users/rtaromax/Documents/cdc/data_per_week/SP_results/SP_results_'+str(i+27)+'.csv', index=False)
    
'''
y_true = np.asarray(list(y_test))
y_pred = np.asarray(list(pred))
score = bio_f_score(y_true, y_pred)
roc_auc_score(y_true, y_pred)
'''