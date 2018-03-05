import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import xgboost as xgb

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load the dataset
df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')

print(df_train.head(), '\n' ,df_test.head())


#Add comment_text in train and test into a big chunk
train_text = df_train['comment_text']
test_text  = df_test['comment_text']
all_text = pd.concat([train_text, test_text])

#Change the whole text words into vector (Word Level)
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=60000)
    #original max_features=10000.
word_vectorizer.fit(all_text)

#Change the train and test text into vector (Word Level)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print(train_word_features.shape[1])
print(test_word_features.shape)

total = train_word_features.shape[1] + 1

train = pd.DataFrame()

len(train_word_features)

A = np.zeros(60000)

for i in range(0, train_word_features.shape[0]):
    a = []
    for j in range(0, train_word_features.shape[1] ):
        a = np.append(a, train_word_features[i, j])
    A = np.vstack([A, a])
print(a)

y_train = df_train.iloc[:, 2:8]
print(y_train.shape)
y1 = y_train['toxic']
y2 = y_train['severe_toxic']
y3 = y_train['obscene']
y4 = y_train['threat']
y5 = y_train['insult']
y6 = y_train['identity_hate']

#######################  XGBOOST  #######################
# Set xgboost parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 4
#params['subsample'] = 0.9
#params['colsample_bytree'] = 0.9

#----1
d_train = xgb.DMatrix(train_features, y1)
d_test = xgb.DMatrix(test_features)

print(d_train)

mdl = xgb.train(params, d_train, 10000, early_stopping_rounds=100, maximize=True, verbose_eval=10)
p_test1 = mdl.predict(d_test)

#----2
d_train = xgb.DMatrix(train_features, y2)

mdl = xgb.train(params, d_train, 10000, early_stopping_rounds=100, maximize=True, verbose_eval=10)
p_test1 = mdl.predict(d_test)
#----3

#----4

#----5

#----6



#------------------ SAVE FILE ---------------------------

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







