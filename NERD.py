'''
Author: Gananath R
NERD: Evolution of Discrete data with Reinforcement Learning
Date: 19/OCT/2019
'''


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


# Loading Reward-Fitness Model
net = Net(seq_len=125, char_len=25)
net.load_state_dict(T.load('model.pth'))

# Load data
df =  np.load('data.npy')

df = pd.DataFrame(df,columns=['SMILES','Labels'])


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

char_max_len = df.SMILES.str.len().max()
char_min_len = df.SMILES.str.len().min()

fake_data = fake_data_gen(df.SMILES,chars,char_min_len-20, char_max_len,5)
df = pd.DataFrame(fake_data,columns=['SMILES','Labels'])

maxX = df.SMILES.str.len().max() 
seed_smiles = df.SMILES.str.ljust(maxX+(125-maxX), fillchar='|')
ts = seed_smiles.str.len().max()

# Parameters
total_population =100
epochs = 500
seed_size = 5
sample_size = int(0.15 * total_population)


# Actor-Critic Model

actor_critic = Net(seq_len=125,char_len=25,actor_critic=True,output_dim=[3,125])

# Optimizer and Loss fuction
opt = T.optim.Adam(actor_critic.parameters(), lr=0.001)

# t1 = T.randn(2,125,25)
# t2 = T.randn(2,1)

# o1,o2,b = actor_critic(t1,t2)
# o1.shape



for i in range(epochs):
    # [smiles,reward,fitness,log_probs1,log_probs2,state_value]
    candidates, action_taken = select_candidates(seed_smiles,ts,char_idx,chars,actor_critic,net,total=total_population)
    
    
    # Update Policy
    policy_loss1 = []
    policy_loss2 = []
    value_loss = []
    total_reward = 0
    smile_reward_fit = []
    for (smi,R,fit,log_p1,log_p2,value) in candidates:
        R =  float(R)
        advantage = R - value.item()
        
        policy_loss1.append(-log_p1*advantage)
        
        policy_loss2.append(-log_p2*advantage)
            
        value_loss.append(F.smooth_l1_loss(value,T.tensor([R])))
            
        total_reward += R
        smile_reward_fit.append([smi,R,fit])

    opt.zero_grad()
    loss = T.stack(policy_loss1).sum() + T.stack(value_loss).sum() + T.stack(policy_loss2).sum()
    loss.backward() 
    opt.step()

    
    # Selecting new seeds 
    df = pd.DataFrame(smile_reward_fit, columns =['SMILES','Reward','Fitness'])
    df[["Reward", "Fitness"]] = df[["Reward", "Fitness"]].apply(pd.to_numeric)
    df.sort_values(['Reward', 'Fitness'], ascending=[False, False],inplace = True)
    if i%10 == 0:
        print("Epoch: "+ str(i) + " Reward: "+ str(total_reward)+" Loss: "+str(round(loss.item(),3)))
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df.head(3))
        print("\n")
    # Select top seeds
    seed_smiles = df.SMILES.head(seed_size).reset_index(drop=True)
    # Random sample from the top seeds
        #seed_smiles = seed_smiles.sample(seed_size).reset_index(drop =True)
