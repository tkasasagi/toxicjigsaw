import pandas as pd

submission = pd.read_csv("hight_of_blend_v2.csv")

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

val_min = 0.00001
val_max = 0.9999
for class_name in class_names:
    train_target = submission[class_name].values

    print(train_target)
    for i in range(len(train_target)):
        if train_target[i] < val_min:
            train_target[i] = 0
        elif train_target[i] > val_max:
            train_target[i] = 1
    
    submission[class_name] = train_target

submission.to_csv('stupidsub3.csv', index=False)