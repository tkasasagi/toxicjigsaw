import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('input/train2.csv')
test = pd.read_csv('input/test2.csv')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=50000)
    #original max_features=10000.
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

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

#max_feature=50000
vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)

vectorizer.fit(all_text)
train_features = vectorizer.transform(train_text)
test_features = vectorizer.transform(test_text)
'''
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
'''

print('Shape of Sparse Matrix: ', train_word_features.shape)
print('Amount of Non-Zero occurences: ', train_word_features.nnz)

sparsity = (100.0 * train_word_features.nnz / (train_word_features.shape[0] * train_word_features.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression

submission = pd.DataFrame.from_dict({'id': test['id']})

for class_name in class_names:
    train_target = train[class_name]
    #train_features, train_target = make_regression(n_features=100000, n_informative=100000, random_state=0, shuffle=False)

    detect_model = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state=0).fit(train_features, train_target)
    
    pred_score = detect_model.predict(test_features)
    submission[class_name] = pred_score
    print('Done')



submission.to_csv('sub.csv', index=False)












