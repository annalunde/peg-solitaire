# aiprog

# Peg solitaire solver using Reinforcement Learning

This is the first of three projects in the AI programming course at NTNU. The group built a general-purpose Actor-Critic Reinforcement Learner and has applied it to different instances of the game Peg Solitaire. The Actor-Critic Reinforcement Learner was built both as a lookup-table and as a neural network.

Figure 1 provides a high-level view of the system design.

![Actor-Critic Reinforcement Learner system](images/td.png){:height="50%" width="50%"}

**File structure:**

- agent
  - actor.py
  - critic.py
  - critic_nn.py
  - critic_dict.py
  - split_gd.py
- environment
  - board.py
  - diamond_board.py
  - enviroment.py
  - triangle_board.py

The configs folder consists of different configs that have been used for the different instances of the game. In main.py it reads in these configs
and starts the whole training loop.

|                        Progression of Learning                         |                                Visualization of Game Play                                 |
| :--------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| ![Actor-Critic Reinforcement Learner progression](images/learning.png) | ![Visualization of game play](https://media.giphy.com/media/2exV3fa4z82ytv9pCf/giphy.gif) |
