
# NERD FOR IMAGES
Trained NERD algorithm to generate images.

![nerd_mnist](https://raw.githubusercontent.com/Gananath/NERD/master/NERD_IMAGES/nerd_mnist.png)

*Numbers at the top of the image is reward/fitness values*

# Notes
I have found some problems with my initial NERD pytorch implementation for sequence generation. The main one was in the neural network for reward fitness function itself which failed to learn anything. Thats why I chose image generation which is well a studied problem and have lots of newtork implementation available in the web. A small dataset of about 200 images were used for training reward fitness model. The trained reward fitness model has a 100 percentage accuracy with very low error values between 0.02 and 0.05.


# Important changes

- Like NERD v2 reward fitness function is a single neural network with a single output which gives reward and fitness values for the problem.

- Removed crossover and deletion process because it was not helping much with the learning.

- Only mutation was kept and its discretized.

- A kernel (m x n) approach was used for mutaion instead of pixel level.

- The actor critic model have control over selecting the region over the images.


