import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import xgboost as xgb

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')

print(df_train.head(), '\n' ,df_test.head())

train_text = df_train['comment_text']
test_text  = df_test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=20000)
    #original max_features=10000.
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])







"""

x_train = xgb.DMatrix(train_features)

y_train = xgb.DMatrix(test_features)


scores = []
val_min = 0.01
val_max = 0.9
submission = pd.DataFrame.from_dict({'id': test['id']})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    pred_score = classifier.predict_proba(test_features)[:, 1]
    print(pred_score)
    for i in range(len(pred_score)):
        if pred_score[i] < val_min:
            pred_score[i] = 0
        elif pred_score[i] > val_max:
            pred_score[i] = 1
    
    submission[class_name] = pred_score

print('Total CV score is {}'.format(np.mean(scores)))


submission.to_csv('sub.csv', index=False)

"""







