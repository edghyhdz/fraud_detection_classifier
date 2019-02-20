import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold as SKF

def eval_input_func(x_,y_,model):

    eval_input_func = tf.estimator.inputs.pandas_input_fn(
          x=x_,
          y=y_,
          batch_size=50,
          num_epochs=1,
          shuffle=False)

    evalmodel = model.evaluate(eval_input_func)

    a = []

    for evalu in evalmodel:
        a.append(evalmodel[evalu])

    return a[0]

def train_model(train_data , valid_data, feat_cols,fold=1):
    #scores_=[]
    hold_score = {}
    train_score = {}

    train_data,valid_data = train_data.reset_index(),valid_data.reset_index()

    train = train_data
    test = valid_data

    X_train_1,y_train_1 = train.drop('isFraud',axis=1),train['isFraud']
    X_test,y_test = test.drop('isFraud',axis=1),test['isFraud'] ##important row

    X = X_train_1
    y = y_train_1
    kf = SKF(n_splits=5, shuffle=True)
    kf.get_n_splits(X,y)
    kfold_train_1=[]
    kfold_test_1=[]


    for train_index, test_index in kf.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        kfold_train_1.append([train_index])
        kfold_test_1.append([test_index])

        #hidden_units=[16,16,16,16,16,16]
    dnn_model = tf.estimator.DNNClassifier(hidden_units=[16,16,16,16],feature_columns=feat_cols,model_dir='/home/edgar/Desktop/Fraud/saved_models_{}/'.format(fold),n_classes=2,optimizer=lambda: tf.train.AdamOptimizer(
        learning_rate=tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=tf.train.get_global_step(),
            decay_steps=5000,
            decay_rate=0.86)))

    for i in range(0,5):

        train = train_data.iloc[list(kfold_train_1[:][i][0])]
        test = valid_data.iloc[list(kfold_test_1[:][i][0])]


        X_train_2,y_train_2 = train.drop('isFraud',axis=1),train['isFraud']
        X_test_2,y_test_2 = test.drop('isFraud',axis=1),test['isFraud']

        input_func = tf.estimator.inputs.pandas_input_fn(x=X_train_2,y=y_train_2,batch_size=800,num_epochs=1500,shuffle=True)


        dnn_model.train(input_fn=input_func,steps=15000)

        hold_score['Kfold_{}_sub_{}'.format(fold,i)] = eval_input_func(x_ = X_train_2, y_ =y_train_2,model = dnn_model)
        train_score['Kfold_{}_sub_{}'.format(fold,i)] = eval_input_func(x_ = X_test, y_ = y_test,model = dnn_model)


    train_score_,hold_score_ = [],[]

    for key in hold_score.keys():
        hold_score_.append(hold_score[key])
        train_score_.append(train_score[key])

    dftocsv = pd.DataFrame({'train_score':train_score_,'hold_score':hold_score_})
    dftocsv.to_csv('scores_{}.csv'.format(fold))

    b = eval_input_func(x_ = X_test, y_ = y_test,model = dnn_model)


    print('\n')
    print('*********************************************************')
    print('*********************************************************')
    print('\n')
    #print(i)
    print('Fold {}, Accuracy: {}'.format((fold), b))
    print('\n')
    print('*********************************************************')
    print('*********************************************************')
    print('\n')
    #scores_.append(b[0])
