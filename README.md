# Porject 1: Details to my Solution


## The Solution

Please visit the **Navigation.ipynb** Jupyter notebook to check my solution. Starting from the block __4. It's Your Turn!__.

In order to find good parameters, I generated the following figure showing the average score over the last 100 scores.
The first row shows a network with two hidden layers of [64,64] followed by [128,128] and in the last row [256,256] neurons and the column with dropout rates of 0.0, 0.25 and 0.5.
The plots itself are showing the average score using Replaybuffer of 100, 10000 and 100000 with updates as shown in the legend. The buffer size of 100 and an update every 100 should represent the use of not Replaybuffer.

![sores](sores.gif)


Depending on the setting, I was able to train the network within 500 episodes to gain an average result of 13 using two hidden layers with each 256 neurons and a replay buffer with a high update rate.

## Learning Algorithm

For this project I used the DeepQNetwork used in the Lunar Landing example as base with a replay buffer.

### hyperparameter and model architectures

The neural network has *two hidden layers* connected with a *ReLU* activation function. For testing, I also tried to use dropouts, but the data suggested that it makes only sense on bigger networks such like two hidden layers with each 265 neurons.

#### Traning

For the training, I used a

* discount factor of GAMMA=0.99
* learning ragte of LR=0.0005
* batch size BATCH_SIZE = 64

## Result

<video width="1780" height="720">
<source src="result.mp4" type="video/mp4">
</video>

## future ideas

* I would like to test networks with more layers of different sizes, but time is currently for me expensive :-).
* Since I used the lunar lander network implantation as base it was easy to use the replay buffer, but It would make sense to test also a implementation without it. Currently I tried only reducing the replay buffer and update rate to mimic a system without replay buffer.
* Tuning GAMMA, LR and BATCH_SIZE was not needed since I received always a score of 13 with one of my solutions, but of course these parameters can be optimized.


## Suggestions for the Exercise

Currently, it is not clear what the feature vector represents and my questions in the forum was not answered. This is sad because it would make sense to get this information to get a better feeling on the problem.
# Install

### Ubuntu 20.04

1. Install anaconda3
2. Create conda enviroment
```
conda create --name udacity
conda activate udacity
cd ...../p1_navigation/
pip3 install -r requirements.txt
```

### Delete python kernel

```
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel
```
 
# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
