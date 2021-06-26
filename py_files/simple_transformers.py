# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 18:52:42 2021

@author: naomi
"""

import pandas as pd
from simpletransformers.classification import ClassificationModel
from train_val_test_data import tt_split


prefix = 'saved_train_test/'


X_train, X_val, y_train, y_val, final_text = tt_split(prefix + 'train.csv',
                             prefix + 'test.csv', 0.2, 42)                         

traindf = pd.DataFrame({'text':X_train.replace(r'\n', ' ', regex=True),
                        'labels':y_train.values})

evaldf = pd.DataFrame({'text':X_val.replace(r'\n', ' ', regex=True),
                       'labels':y_val.values})


# Create a TransformerModel
model = ClassificationModel('roberta', 'roberta-base', '')

# Train the model
model.train_model(traindf)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(evaldf)