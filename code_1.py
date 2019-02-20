import numpy as np
import pandas as pd
import tensorflow as tf
from train_model import *


fold = 1

train_fold = pd.read_csv('test_fold_{}.csv'.format(fold),index_col=0)
test_fold = pd.read_csv('test_fold_{}.csv'.format(fold),index_col=0)


cols = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest'] ##

numerical_cols = {}
for i in cols:
    numerical_cols[i] = tf.feature_column.numeric_column(i)
feat_cols = [numerical_cols[things] for things in numerical_cols]

train_model(train_data = train_fold, valid_data = test_fold,feat_cols=feat_cols, fold=fold)
