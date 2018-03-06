# All credits goes to original authors.. Just another blend...
import pandas as pd
from sklearn.preprocessing import minmax_scale
sup = pd.read_csv('sub/hight_of_blend_v2.csv')
allave = pd.read_csv('sub/submit_cnn_avg_3_folds.csv')
gru = pd.read_csv('sub/submission-tuned-LR-01.csv')

blend = allave.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.2*minmax_scale(allave[col].values)+0.6*minmax_scale(gru[col].values)+0.2*minmax_scale(sup[col].values)
print('stay tight kaggler')
blend.to_csv("crazy3.csv", index=False)

