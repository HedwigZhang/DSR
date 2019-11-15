import pickle
import pandas as pd

f = open('resnet50_2_DS_with_res_no_bias.pkl', 'rb')
data = pickle.load(f)
f.close()

acc = data['acc']
loss = data['loss']
val_acc = data['val_acc']
val_loss = data['val_loss']
time = data['elapsed']
top_k_acc = data['top_k_categorical_accuracy']
val_top_k_acc = data['val_top_k_categorical_accuracy']

df = pd.DataFrame(data={'loss':loss, 'acc':acc, 'top_k_acc':top_k_acc, 'val_loss':val_loss, 'val_acc':val_acc, 'val_top_k_acc':val_top_k_acc, 'time': time})
# df = pd.DataFrame(data={'loss':loss, 'acc':acc, 'val_loss':val_loss, 'val_acc':val_acc, 'time': time})
df.to_csv('resnet50_2_DS_with_res_no_bias.csv')