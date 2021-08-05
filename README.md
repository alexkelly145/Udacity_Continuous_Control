# Udacity_Continuous_Control
Project 2: Continuous Control

The environment I am using for this project is the Reacher environment created by Unity. 

The environment contains 20 double jointed arms. The objective of the environment is to keep the hands of these arms in a target location for as many time steps as possible. 
The environment is considered solved when an average reward of +30 across 100 episodes is achieved for all 20 arms.
The agents receive a reward of +0.1 if the arm stays in the target location.

The observation space of the state contains 33 variables which contain information about the position, velocity, angular velocities and rotation of the arm. 
Each action is a vector of 4 numbers that are values between -1 and 1. 


Getting Started

The environment is Windows (64-bit) reacher environment: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip

Setting up the environment

1. First we need to create and activate the environment with Python 3.6

	conda create --name drlnd python=3.6 
	activate drlnd

2. Next download the enviroment dependencies with the requirements.txt file

	pip install -r requirements.txt

3. Clone the Github repository 

	git clone https://github.com/udacity/deep-reinforcement-learning.git
	cd deep-reinforcement-learning/python
	pip install .

4. Create an IPython kernel for the drlnd environment

	python -m ipykernel install --user --name drlnd --display-name "drlnd"

5. Make sure drlnd is selected and drlnd appears in the top right corner instead of Python when inside the notebook

Running the code

There are 4 files needed to run the code:

	1. main.ipynb
	2. ddpg_agent.py
	3. replay.py
	4. model.py

Make sure all these files are in the same directory.

Open the ddpg_main.ipynb notebook, the hyperparameters for the DDPG agent can be changed in cell 4.
