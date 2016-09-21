#sequential-decision-making-for-a-humanod-robot
This repository contains source code, LaTeX templates, and figures of the our on-going study whic will be appeared on [Humanoids 2016](http://humanoids2016.org/). Note that the current state of the repository does not contain finalized source code.

> .. one must build a thing to truly understand it.

The pre-final abstract of the submitted paper is blockquoted below;

> Certain emotions and moods can be manifestations of complex and costly neural computations that our brain wants to avoid. Instead of reaching an optimal decision based on the facts, we find it often easier and sometimes more useful to rely on hunches. In this work, we extend a previously developed model for such a mechanism where a simple neural associative memory was used to implement a visual recall system for a humanoid robot. In the model, the changes in the neural state consume (neural) energy, and to minimize the total cost and the time to recall a memory pattern, the robot should take the action that will lead to minimal neural state change. To do so, the robot needs to learn to act rationally, and for this, it has to explore and find out the cost of its actions in the long run. In this study, a humanoid robot (iCub) is used to act in this scenario. The robot is given the sole action of changing his gaze direction. By reinforcement learning (RL) the robot learns which state-action pair sequences lead to minimal energy consumption. More importantly, the reward signal for RL is not given by the environment but obtained internally, as the actual neural cost of processing an incoming visual stimuli. The results indicate that reinforcement learning with the internally generated reward signal leads to non-trivial behaviours of the robot which might be interpreted by external observers as the robot's 'liking' of a specific visual pattern, which in fact emerged solely based on the neural cost minimization principle. 
