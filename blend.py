
import pandas as pd
import numpy as np

sub1 = pd.read_csv('sub/blend_it_all.csv')
sub2 = pd.read_csv('sub/newsub2.csv')

#grucnn = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv')
#gruglo = pd.read_csv("../input/pooled-gru-glove-with-preprocessing/submission.csv")
#ave = pd.read_csv("../input/toxic-avenger/submission.csv")
#supbl= pd.read_csv('../input/blend-of-blends-1/superblend_1.csv')
#best = pd.read_csv('../input/toxic-hight-of-blending/hight_of_blending.csv')
#lgbm = pd.read_csv('../input/lgbm-with-words-and-chars-n-gram/lvl0_lgbm_clean_sub.csv')
#wordbtch = pd.read_csv('../input/wordbatch-fm-ftrl-using-mse-lb-0-9804/lvl0_wordbatch_clean_sub.csv')
#tidy = pd.read_csv('../input/tidy-xgboost-glmnet-text2vec-lsa/tidy_glm.csv')
#fast = pd.read_csv('../input/pooled-gru-fasttext-6c07c9/submission.csv')
#bilst = pd.read_csv('../input/bidirectional-lstm-with-convolution/submission.csv')
#oofs = pd.read_csv('../input/oof-stacking-regime/submission.csv')
#corrbl = pd.read_csv('../input/another-blend-tinkered-by-correlation/corr_blend.csv')
#rkera = pd.read_csv('../input/why-a-such-low-score-with-r-and-keras/submission.csv')

#b1 = best.copy()
#col = best.columns

#col = col.tolist()
#col.remove('id')
#for i in col:
#    b1[i] = (2 * fast[i]  + 2 * gruglo[i] + grucnn[i] * 4 + ave[i] + supbl[i] * 2 + best[i] * 4 +  wordbtch[i] * 2 + lgbm[i] * 2 + tidy[i] + bilst[i] * 4 + oofs[i] * 5 + corrbl[i] * 4) /  33
    
#b1.to_csv('blend_it_all.csv', index = False)

b1 = sub1.copy()
col = sub1.columns

col = col.tolist()
col.remove('id')
for i in col:
    b1[i] = ( sub1[i] * 8 -  sub2[i] * 2) /  6
    
b1.to_csv('myblend.csv', index = False)
