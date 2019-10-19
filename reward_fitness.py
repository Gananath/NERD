'''
Author: Gananath R
NERD: Evolution of Discrete data with Reinforcement Learning
Date: 19/OCT/2019
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
try:
    from torchsummary import summary
except:
    pass

from tqdm import tqdm
from drugai import *
import pandas as pd
import numpy as np
import torch as T 
import random
import os



# Reproducibility
seed = 2019
random.seed(seed)
T.manual_seed(seed)
np.random.seed(seed)

#epoch
epoch = 200
#batch size
bs = 30


# read file into pandas from the working directory
if not os.path.exists('data.npy'):
    df = pd.read_csv('stahl.csv')
    df = pd.DataFrame(df.SMILES,columns=['SMILES'])
    df['Labels'] = 1
    char_max_len = df.SMILES.str.len().max()
    char_min_len = df.SMILES.str.len().min()
    print("Mean: "+str(df.SMILES.str.len().mean())+\
    " Max: "+str(char_max_len)+\
    " Min: "+str(char_min_len))

    # Padding smiles to same length by adding "|" at the end of smiles
    maxX = df.SMILES.str.len().max() 
    df.SMILES = df.SMILES.str.ljust(maxX+(125-maxX), fillchar='|')
    ts = df.SMILES.str.len().max()
    print ("ts={0}".format(ts))
    # CharToIndex and IndexToChar functions
    chars = sorted(list(set("".join(df.SMILES.values.flatten()))))
    print('total chars:', len(chars))

    # fake data 1
    fake_data = np.copy(df.SMILES.values)
    fake_data = shuffle3D(fake_data)
    fake_data = np.vstack((fake_data,np.zeros(len(fake_data)))).T
    fake_data = pd.DataFrame(fake_data,columns=['SMILES','Labels'])
    df = df.append(fake_data)

    # fake data 2
    fake_data = fake_data_gen(df.SMILES, chars,char_min_len-20, char_max_len+20,int(len(df)/3))
    fake_data = pd.DataFrame(fake_data,columns=['SMILES','Labels'])
    print("Length of fake: ",fake_data.SMILES.str.len().max() )
    df = df.append(fake_data)
        
    df = df.reset_index()
    
    del fake_data, df['index']
    # Save
    np.save('data.npy',arr = df[['SMILES','Labels']].values,allow_pickle=True)
    
else:
    df =  np.load('data.npy')


df = pd.DataFrame(df,columns=['SMILES','Labels'])
#df = df.reindex(np.random.permutation(df.index))

# Padding smiles to same length by adding "|" at the end of smiles
maxX = df.SMILES.str.len().max() 
X = df.SMILES.str.ljust(maxX, fillchar='|')
ts = X.str.len().max()
print ("ts={0}".format(ts))
# CharToIndex and IndexToChar functions
chars = sorted(list(set("".join(X.values.flatten()))))
print('total chars:', len(chars))

char_idx = dict((c, i) for i, c in enumerate(chars))
idx_char = dict((i, c) for i, c in enumerate(chars))

X_dash = dimY(X, ts, char_idx, chars)
y = one_hot(df.Labels.values.astype(int),2).astype(float)


# Neural network model
net = Net(seq_len=X_dash.shape[1],char_len = X_dash.shape[2])

# Optimizer and Loss fuction
optimizer = T.optim.Adam(net.parameters(), lr=0.001)
loss_function = T.nn.BCELoss()

out = net(T.randn(1,125,25),None)

#if summary:
#    summary(net,[(125,25),(1,)])




pbar = tqdm(range(epoch))
score = 0
prev_loss = 10
for e in pbar:
    X_train, X_test, y_train, y_test = train_test_split(X_dash, y, test_size=0.10,shuffle=True)
    
    for i in range(0,len(X_train),bs):
        batch_X = T.from_numpy(X_train[i:i+bs]).float()
        batch_y = T.from_numpy(y_train[i:i+bs]).float()
        
        net.zero_grad()
        
        outputs = net(batch_X,None)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Validation
    y_pred = np.argmax(net(T.from_numpy(X_test).float(),None).detach().numpy(),axis=1)
    
    y_test = np.argmax(y_test,axis=1)
    
    acc = accuracy_score(y_test, y_pred, normalize=False)
    pbar.set_description("Acc: %f Loss %f" %(acc,loss.item()))
    if acc >= score and loss.item() < prev_loss:
        T.save(net.state_dict(), 'model.pth')
        score = acc
        prev_loss = loss.item()



'''
#Testing

net.load_state_dict(T.load('model.pth'))
_, X_test, _, y_test = train_test_split(X_dash, y, test_size=0.10,shuffle=True)

y_pred = np.argmax(net(T.from_numpy(X_test).float()).detach().numpy(),axis=1)
y_test = np.argmax(y_test,axis=1)

acc = accuracy_score(y_test, y_pred, normalize=False)
print(acc)
'''

