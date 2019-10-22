from collections import namedtuple
from pysmiles import read_smiles
import pandas as pd
import numpy as np
import torch as T 
import random
import os

F = T.nn.functional

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def fake_data_gen(df,chars,char_min_len,char_max_len,total=100):
    arr= []
    for i in range(total):
        rand_var = random.randint(char_min_len, char_max_len)
        string = ''.join([random.choice(chars) for _ in range(rand_var)])
        arr.append([string,-1])
    return np.array(arr)

# time step addtition to target
def dimY(Y, ts, char_idx, chars):
    temp = np.zeros((len(Y), ts, len(chars)), dtype=np.bool)
    for i, c in enumerate(Y):
        for j, s in enumerate(c):
            # print i, j, s
            temp[i, j, char_idx[s]] = 1
    return np.array(temp)


def seq_txt(y_pred, idx_char):
    newY = []
    for i, c in enumerate(y_pred):
        newY.append([])
        for j in c:
            newY[i].append(idx_char[np.argmax(j)])
    return np.array(newY)

# joined smiles output
def smiles_output(y_pred):
    smiles = np.array([])
    for i in y_pred:
        j = ''.join(i)
        smiles = np.append(smiles,j)
    return smiles

class Net(T.nn.Module):
    def __init__(self,seq_len = 25, char_len = 25,actor_critic=False,output_dim = [2,125]):
        super().__init__()
        self.c1 = T.nn.Conv1d(seq_len, char_len, kernel_size=2)
        self.p1 = T.nn.AvgPool1d(2)
        self.c2 = T.nn.Conv1d(char_len, int(char_len/2), kernel_size=2)
        self.p2 = T.nn.AvgPool1d(2)
        self.c3 = T.nn.Conv1d(char_len, int(char_len/4), kernel_size=2)
        self.p3 = T.nn.AvgPool1d(2)
        self.drop = T.nn.Dropout(p=0.3)
        self.inp = T.nn.Linear(1,60)
        self.fc1 = T.nn.Linear(120, 30)
        self.fc1_1 = T.nn.Linear(60, 30)
        self.fc2 = T.nn.Linear(30, 10)
        self.fc3 = T.nn.Linear(10, output_dim[0])
        self.fc4 = T.nn.Linear(10, output_dim[1])
        self.value_layer= T.nn.Linear(10, 1)
        self.actor_critic = actor_critic
    def forward(self, x, y):
        x = self.conv_block(x, y)
        #print(x.shape)
        if self.actor_critic:
            x = x.view(-1, 120)
            x = F.relu(self.fc1(x))
        else:
            x = x.view(-1, 60)
            x = F.relu(self.fc1_1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        if self.actor_critic:
            action_layer1 = F.softmax(self.fc3(x), dim=1)
            action_layer2 = F.softmax(self.fc4(x), dim=1)
            return action_layer1, action_layer2 , self.value_layer(x)
        else:
            return self.fc3(x)
    def conv_block(self,x, y):
        x = F.relu(self.c1(x))
        x = F.relu(self.p1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.p2(x))
        x = self.drop(x)
        if self.actor_critic:
            sec_inp = F.relu(self.inp(y))
            x = x.view(x.size(0),-1)
            x = T.cat((x,sec_inp),dim=1)
        return x


def shuffle3D(arr):
    new_arr =[]
    for i,a in enumerate(arr):
        new_arr.append(''.join(random.sample(a,len(a))))
    return np.array(arr)

def mutation(X,chars,loc=25):
    arr =[]
    for i in X:
        smiles = list(i)
        smiles[loc] = random.choice(chars)
        smiles = "".join(smiles)
        arr.append(smiles)
    return np.array(arr)

def deletion(X,loc=25):
    arr = []
    for i in X:
        smiles = list(i)
        del smiles[loc]
        smiles.insert(len(smiles),'|')
        smiles = "".join(smiles)
        arr.append(smiles)
    return np.array(arr)


def crossover(X,loc=25):
    chromosome1 = X[0]
    chromosome2 = X[1]
    
    gene11, gene12 = chromosome1[:loc] , chromosome1[loc:]
    gene21, gene22 = chromosome2[:loc] , chromosome2[loc:]
    
    return np.array([gene11+gene22, gene21+gene12])


def select_candidates(X,ts,char_idx,chars,actor_critic,net,total=100):
    SavedAction = namedtuple('SavedAction', ['smi','rew_fit','log_p1','log_p2', 'value'])
    action_taken=[]
    saved_actions =[]
    for i  in range(int(total/2)):
        rand_rows = np.random.randint(0,X.shape[0],2)
        state = X[rand_rows]
        state3d = dimY(state, ts, char_idx, chars)
        state3d = state3d.astype('float')
        
        t_state = T.from_numpy(state3d).float()
        z_value = T.randn(2,1)
        probs1, probs2, state_value = actor_critic(t_state,z_value)
        # First action probs
        m = T.distributions.Categorical(probs1)
        action1 = m.sample()
        log_probs1 = m.log_prob(action1).unsqueeze(1)
        # Second action probs
        n = T.distributions.Categorical(probs2)
        action2 = n.sample()
        log_probs2 = n.log_prob(action2).unsqueeze(1)
        # if any of the two actions is a crossover then it would be given preference
        if 2 in action1:
            new_genes = crossover(state.values,action2.max().item())
            action_taken.append('crossover')
        else:
            new_genes = np.array([])
            for j,a in enumerate(action1):
                if a.item() == 0:
                    new_genes = np.append(new_genes,mutation([X[j]],chars,action2[j].item()))
                    action_taken.append("mutation")
                elif a.item() == 1:
                    new_genes = np.append(new_genes,deletion([X[j]],action2[j].item()))
                    action_taken.append("deletion")
        
        smiles_reward_fit = get_reward_fitness(state,new_genes,ts,char_idx,chars,net)
        
        #[smiles reward,fitness,log_probs1,log_probs2,state_value]
        for i, j,k,l in zip(smiles_reward_fit,log_probs1,log_probs2,state_value):
            sm = i[0]
            rew_fit = i[1]
            saved_actions.append(SavedAction(sm,rew_fit,j,k,l))
    
    return saved_actions, action_taken


def get_reward_fitness(state,X,ts,char_idx,chars,net):
    X_seed = dimY(X, ts, char_idx, chars)
    out = net(T.from_numpy(X_seed).float(),None).detach().numpy()
    
    
    # [smiles,reward+fitness]
    arr = np.hstack((X.reshape(len(X),1),out))
    
    return arr
