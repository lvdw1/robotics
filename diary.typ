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

== Creating second link
=== Prep
+ Extra branch on git
+ Created "simulation_pendulum2.yaml" and "furuta_pendulum2.urdf". Both with the help of Gemini 2.5 Pro
+ Created "forward_dynamics2.casadi" inspired by code from notebook from class (Lecture 2 I think)
+ Tried to run the "model_tester", but some issues with dimensions
  - So, started to look at furuta_torque_env
    + ```python q_init = np.array([0, 0, 0])```
    + ```python self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])```
    + ```python 
    q_sym = ca.MX.sym("q", 3) \
    dq_sym = ca.MX.sym("dq", 3)
      ```
    + ```python self.viz.display(self.x[:2])```
    + ```python
    if swingup:
            self.init_qpos = np.array([0.0, 0.0, 0.0])
        else:
            # Start at upward position
            self.init_qpos = np.array([0.0, np.pi, np.pi])

        self.init_qvel = np.array([0.0, 0.0, 0.0])

        self.pendulum_sim.x = np.array([self.init_qpos[0], self.init_qpos[1], self.init_qpos[2], self.init_qvel[0], self.init_qvel[1]], self.init_qvel[2])
        self.qpos = np.array([
            self.pendulum_sim.x[0],
            self.pendulum_sim.x[1],
            self.pendulum_sim.x[2]
        ])
        self.qvel = np.array([
            self.pendulum_sim.x[3],
            self.pendulum_sim.x[4],
            self.pendulum_sim.x[5],
        ])
        ```
    + ```python
    self.observation_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(9,),
        dtype=np.float64,
    )
    ```
    + ```python
    def step(self, action):
        u = action[0]*self._max_torque_joint0 
        self.pendulum_sim.step(u, self.dt)
        self.qpos = np.array([
            self.pendulum_sim.x[0],
            self.pendulum_sim.x[1], 
            self.pendulum_sim.x[2]
        ])
        self.qvel = np.array([
            self.pendulum_sim.x[4],
            self.pendulum_sim.x[4],
            self.pendulum_sim.x[5]
        ])
    ```
    + ```python
    def reset(self, seed=None, options=None):
        qpos = self.init_qpos
        qvel = self.init_qvel

        self.pendulum_sim.x = np.array([qpos[0], qpos[1], qpos[2], qvel[0], qvel[1],qvel[2]])
        self.qpos = np.array([qpos[0], qpos[1], qpos[2]])
        self.qvel = np.array([qvel[0], qvel[1], qpos[3]])
    ```
    + ```python
    obs = np.array(
        [
            np.sin(self.qpos[0]),
            np.cos(self.qpos[0]),
            np.sin(self.qpos[1]),
            np.cos(self.qpos[1]),
            np.sin(self.qpos[2]),
            np.cos(self.qpos[2]),
            self.qvel[0] / self._max_velocity_joint0,
            self.qvel[1] / self._max_velocity_joint1,
            self.qpos[0] / self._max_angle_joint0,
            # self.qpos[1] / self._max_angle_joint1,
        ]
    )
    ```

