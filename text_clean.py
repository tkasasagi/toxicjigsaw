import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('input/train2.csv')
test = pd.read_csv('input/test2.csv')

import re
#re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
#def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def cleantext(text):
    text = re.sub(r'[^\w\s]','',text)
    result = ''.join([i for i in text if not i.isdigit()])
    #return re_tok.sub(r' \1 ', result)
    return result
    
newtrain = []
newtest  = []
for i in range(len(train['comment_text'])):
    result = cleantext(train['comment_text'][i])
    newtrain.append(result) 
    
for i in range(len(test['comment_text'])):
    result = cleantext(test['comment_text'][i])
    newtest.append(result)   
    
all_text = newtrain + newtest   
    
#train_text = train['comment_text']
#test_text = test['comment_text']
#all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    min_df=2, max_df=0.8, #1, 0.8
    use_idf=1, smooth_idf=1,
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    lowercase = False,
    max_features=50000) 

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(newtrain)
test_word_features = word_vectorizer.transform(newtest)


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    min_df=2, max_df=0.8,
    use_idf=1, smooth_idf=1, 
    stop_words='english',
    ngram_range=(2, 7),
    lowercase = False,
    max_features=100000)

#max_feature=50000
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(newtrain)
test_char_features = char_vectorizer.transform(newtest)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


print(train_features.dtype)
scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='newton-cg')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('newsub5.csv', index=False)

