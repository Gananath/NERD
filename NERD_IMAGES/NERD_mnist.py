from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch as T 
import pickle
import random
import os
F = T.nn.functional 

# Reproducibility
seed = 2019
random.seed(seed)
T.manual_seed(seed)
np.random.seed(seed)
print(T.__version__)
#%matplotlib inline


class Actor_Critic(T.nn.Module):
    def __init__(self):
        super(Actor_Critic, self).__init__()
        self.conv1 = T.nn.Conv2d(1, 20, 5,1)
        self.conv2 = T.nn.Conv2d(20, 50, 5,1)
        self.dropout = T.nn.Dropout2d(0.5)
        self.inp = T.nn.Linear(1,60)
        self.fc1 = T.nn.Linear(860, 500)
        self.fc2 = T.nn.Linear(500, 10)
        self.fc3 = T.nn.Linear(500, 28)
        self.fc4 = T.nn.Linear(500, 28)
        #self.fc5 = T.nn.Linear(500, 1)
        self.fc6 = T.nn.Linear(500, 1)
    def forward(self, x, y):
        sec_inp = F.relu(self.inp(y))
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = T.cat((x,sec_inp),dim=1)
        x = F.relu(self.fc1(x))
        action = F.softmax(self.fc2(x), dim=1)
        row_action = F.softmax(self.fc3(x), dim=1)
        column_action = F.softmax(self.fc4(x), dim=1)
        #mutation value sets mutation values for images
        #mutation_value = T.sigmoid(self.fc5(x))
        value_layer = self.fc6(x)
        #action, row_action, column_action, mutation_value,value_layer
        return action, row_action, column_action,value_layer



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


def mutation(state,row,column,kernel=3,action=None):
    shp = state[row:row+kernel,column:column+kernel].shape
    for a in range(10):
        if a == action:
            mut_value = np.random.randint(a*10,a*10+10,(shp[0],shp[1]))/100.0
    state[row:row+kernel,column:column+kernel] = mut_value
    return state


def select_candidates(X,net,actor_critic,total_population=100):
    SavedAction = namedtuple('SavedAction', ['States','RF','action','row','column', 'value'])
    saved_actions =[]
    action_taken=[]
    for i in range(int(total_population/2)):
        rand_idx = np.random.randint(0,len(X),2)
        
        # randomly select two states
        state = X[rand_idx]
        z_value = T.rand(2,1)
        
        action_probs, row_probs, column_probs,  ac_value = actor_critic(T.from_numpy(state).float(),z_value)
        
        #genetic  algorithm action
        m = T.distributions.Categorical(action_probs)
        actions = m.sample()
        action_log_probs = m.log_prob(actions).unsqueeze(1)
        
        # row action
        m = T.distributions.Categorical(row_probs)
        row_actions = m.sample()
        row_log_probs = m.log_prob(row_actions).unsqueeze(1)
        
        # column action
        m = T.distributions.Categorical(column_probs)
        column_actions = m.sample()
        column_log_probs = m.log_prob(column_actions).unsqueeze(1)
        
        #taking actions accroding to actor critic model
        new_state = []
        for j,a in enumerate(actions):
            r = row_actions[j].item()
            c = column_actions[j].item()
            state[j][0] = mutation(state[j][0],r,c,action=a)
            new_state.append(state[j])
            action_taken.append(a.item())
            
           
        new_state = np.array(new_state)
        
        # calculating fitness/reward from reward_fitness function
        fit_reward_value = net(T.from_numpy(new_state).float()).detach().numpy()
        
        # Saving torch varibles for backprop
        for s,rf,a,r,c,val in zip(new_state,fit_reward_value,action_log_probs,row_log_probs,column_log_probs,ac_value):
            saved_actions.append(SavedAction(s,rf,a,r,c,val))
    return saved_actions, action_taken


# Parameters
total_population =100
epochs = 20000
seed_size = 5
sample_size = int(0.15 * total_population)


net = Reward_Fitness_Model()
net.load_state_dict(T.load('model_mnist.pth'))

actor_critic = Actor_Critic()
# Optimizer 
opt = T.optim.Adam(actor_critic.parameters(), lr=0.0001)

seed_images = np.random.rand(15,1,28,28)

pbar = tqdm(range(epochs))
for e in pbar:
    candidates, action_taken = select_candidates(seed_images,net,actor_critic,total_population)
    
    action_policy_loss =[]
    row_policy_loss = []
    col_policy_loss = []
    value_loss = []
    total_reward = []
    states = []
    reward_fit = np.array([])
    for (s,rf,act_logprobs,row_logprobs,col_logprobs,val) in candidates:
        #advantage = reward - value
        advantage = rf[0] - val.item()
        action_policy_loss.append(-act_logprobs*advantage)
        row_policy_loss.append(-row_logprobs*advantage)
        col_policy_loss.append(-col_logprobs*advantage)
        value_loss.append(F.smooth_l1_loss(val,T.tensor(rf)))
        total_reward += rf[0]
        states.append(s)
        reward_fit = np.append(reward_fit,rf[0])
    
    opt.zero_grad()
    
    loss = T.stack(action_policy_loss).sum() + T.stack(value_loss).sum() + T.stack(row_policy_loss).sum() + T.stack(col_policy_loss).sum()
    
    loss.backward() 
    opt.step()
    
    #selectiong new seeds
    
    idx = np.argsort(reward_fit)[::-1][:sample_size]
    states = np.array(states)
    seed_images = states[idx]
    reward_fit = reward_fit[idx]
    pbar.set_description(" Loss:%f " %(loss.item()))
    if e%50 ==0:
        unique, counts = np.unique(action_taken, return_counts=True)
        print(dict(zip(unique, counts)))
        T.save(actor_critic.state_dict(), 'acmodel_mnist.pth')
        unique, counts = np.unique(action_taken, return_counts=True)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(1, 11):
            ax = fig.add_subplot(2, 5, i)
            ax.axis('off')
            ax.imshow(np.squeeze(seed_images[i]))
            ax.set_title(str(round(reward_fit[i],3)))
    
        plt.show()
        plt.close()


