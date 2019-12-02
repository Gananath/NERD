
# NERD FOR IMAGES
Trained NERD(v2) algorithm to generate images.


# Notes
I have found some problems with my initial NERD pytorch implementation for sequence generation. The main one was in the neural network for reward fitness function itself which failed to learn anything. Thats why I chose image generation which is a well studied system.


# Important changes

- Like NERD v2 reward fitness function is a single neural network with a single output which gives reward and fitness values for the problem.

- Removed crossover and deletion process because it was not helping much with the learning.

- Only mutation was kept and its discretized.

- A kernel (m x n) approach was used for mutaion instead of pixel level.

- The actor critic model have control over selecting the region over the images.


