# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:29:34 2021

@author: naomi
"""

import pandas as pd
from sklearn.model_selection import train_test_split



def train_val_test_split(traincsv, testcsv, test_plus_val_size=0.4, 
                         random_state=53):
    #Take 'train.csv' and 'test.csv' file and create
    #train, validation, test data sets and returns the numpy arrays of them
    train = pd.read_csv(traincsv, index_col=0)
    test = pd.read_csv(testcsv, index_col=0)

    X_partial_train, X_val, y_partial_train, y_val = train_test_split(train['text'], 
                            train['target'], test_size = test_plus_val_size, 
                            random_state=random_state)
    val_size = int(len(X_val)/2)
    X_partial_val = X_val.iloc[:val_size]
    y_partial_val = y_val.iloc[:val_size]
    X_test = X_val.iloc[val_size:]
    y_test = y_val.iloc[val_size:]

    X_final_test = pd.Series(index=test.index, data=test.text.values)   
    
    return X_partial_train, X_partial_val, X_test, y_partial_train, \
           y_partial_val, y_test, X_final_test
           
def tt_split(trainpath, testpath, test_size=0.2, random_state=123):
    #Take 'train.csv' and 'test.csv' file and create
    #train, validation, test data sets and returns the numpy arrays of them
    train = pd.read_csv(trainpath, index_col=0)   
    train_text = train.loc[:,'text']
    train_target = train.loc[:,'target']
    test = pd.read_csv(testpath, index_col=0)
    final_text = test.loc[:,'text']
    X_train, X_val, y_train, y_val = train_test_split(train_text, 
                            train_target, test_size = test_size, 
                            random_state=random_state)
    

    return X_train, X_val, y_train, y_val, final_text              
           

"""
X_train, X_val, X_test, y_train, \
        y_val, y_test, X_final_test = \
        train_val_test_split('train.csv', 'test.csv', 0.3, random_state=123) 
        
  
"""        