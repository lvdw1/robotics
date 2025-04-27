#show link:underline
#show link:set text(fill:blue)

#align(center)[= Robotics]
\
\
== Base question
=== First tests
+ Training with the same reward function, but starting from downwards for swingup
  - After $10^6$ steps, he was swinging and trying to stay upwards, but not really able to stay there
  - After $3.10^6$ steps, still didn't find it. Of course, slowed down even more, but not able to stay there
+ Modifying the reward function, based on #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10234431")[paper]
  - Using cosine near upward position ($theta = pi$ for us)
  $
  & -7 - cos(theta) + (1-cos(theta))^3 & "if" cos(theta) < -0.76\
  & -0.2   & "else "
  $
  - Works way better: reward is more "extreme" near goal state
  - Also independent of (physical) parameters
  - saved_model_path = 'furuta_pendulum_tensorboard/furutaTorque__ppo_continuous_action__42__1745760168/ppo_continuous_action.cleanrl_model'
=== Todos
+ To prove that more "extreme" reward functions near center is better: increase power of base function from 2 to $arrow.r$ 10
+ Experiment with function that rewards energy (based on GPTo3)
  - Generate energy until you have enough to do the swing-up
    - Amount of energy is dependent on the system parameters
  - From then on, reward being close to goal state 

  == Domain randomisation
  === First implementation
  + Spawn the different environments with 10% noise on every parameter defined in the .yml 
    - This is a nice start, but you only have 25 different environments
    - Would be nicer if new parameters are used after every episode termination
    - This posed no challenge for PPO (meaning that he found a policy that was already semi-robust to slight parameter changes)
