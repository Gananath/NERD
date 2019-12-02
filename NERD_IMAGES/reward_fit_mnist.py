from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import mnist


try:
    from torchsummary import summary
except:
    pass

from tqdm import tqdm
#from drugai import *
import pandas as pd
import numpy as np
import torch as T 
import random
import pickle
import os

F = T.nn.functional 

# Reproducibility
seed = 2019
random.seed(seed)
T.manual_seed(seed)
np.random.seed(seed)


class Reward_Fitness_Model(T.nn.Module):
    def __init__(self):
        super(Reward_Fitness_Model, self).__init__()
        self.conv1 = T.nn.Conv2d(1, 20, 5,1)
        self.conv2 = T.nn.Conv2d(20, 50, 5,1)
        self.dropout = T.nn.Dropout2d(0.5)
        self.fc1 = T.nn.Linear(4*4*50, 500)
        self.fc2 = T.nn.Linear(500, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#epoch
epoch = 500
#batch size
bs = 30

net = Reward_Fitness_Model()

if os.path.exists('model_mnist.pth'):
    net.load_state_dict(T.load('model_mnist.pth'))

loss_function = T.nn.MSELoss()
optimizer = T.optim.Adam(net.parameters(), lr=0.0001)



if not os.path.exists('data.pk1'):
    num_images = 100
    cwd = os.getcwd()
    mnist.temporary_dir = lambda: cwd
        
    x_test = mnist.test_images()
    # normalizing 0 and 1
    x_test = x_test/255
    #y_test = mnist.test_labels()
    
    x_test = x_test[0:num_images]
    y_test = np.ones((num_images,1))
    
    
    x_fake = np.random.rand(num_images,28,28)
    y_fake = np.ones((num_images,1))*-1
    
    X = np.vstack((x_test,x_fake))
    y = np.vstack((y_test,y_fake))
    
    ''''
plt.imshow(X[0][0])
plt.show()
plt.close()
    '''
    data =[X,y]
    
    del x_test,y_test, x_fake,y_fake,X,y
    
    with open('data.pk1','wb') as f: pickle.dump(data, f)
else:
    with open('data.pk1','rb') as f: data = pickle.load(f)

X, y = data

X = np.expand_dims(X,1)

pbar = tqdm(range(epoch))


for e in pbar:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,shuffle=True)
    for i in range(0,len(X_train),bs):
        batch_X = T.from_numpy(X_train[i:i+bs]).float()
        batch_y = T.from_numpy(y_train[i:i+bs]).float()
        
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    with T.no_grad():
        out = net(T.from_numpy(X_test).float())
    pred = np.where(out.numpy()>0,1,-1)
    score = accuracy_score(y_test,pred)
    pbar.set_description(" Loss:%f Accuracy:%f" %(loss.item(),score))
    if e%25==0:
        T.save(net.state_dict(), 'model_mnist.pth')


net.load_state_dict(T.load('model_mnist.pth'))


with T.no_grad():
    out = net(T.from_numpy(X_test).float())

pred = np.where(out.numpy()>0,1,-1)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 11):
    ax = fig.add_subplot(2, 5, i)
    ax.axis('off')
    ax.imshow(np.squeeze(X_test[i]))
    ax.set_title(str((int(y_test[i][0]),int(pred[i][0]))))

plt.show()
plt.close()


