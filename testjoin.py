import pandas as pd
import numpy as np

df_train = pd.read_csv('input/test2.csv')

df_train.head()

df_train.drop('length', axis = 1, inplace = True)


text = df_train['comment_text'][0]

import re


def jointext(text):
    text = re.sub('\', \'', ' ', text)
    text = re.sub('\'', ' ', text)                
    text = re.sub('\"', ' ', text) 
    text = re.sub('\[', ' ', text) 
    text = re.sub('\]', ' ', text) 
    text = re.sub('\,', '', text) 
    text = re.sub('=', '', text) 
    text = re.sub(':', '', text) 
    return text

df_train['comment_text'] = df_train['comment_text'].apply(jointext)


df_train.to_csv('test3.csv', index = False)