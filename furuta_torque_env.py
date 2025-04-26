import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

import pinocchio as pin
from pinocchio.utils import *
from pinocchio.visualize import MeshcatVisualizer as PMV

import time

class FurutaPendulumSimulator:
    def __init__(self, urdf_model_path, forward_dynamics_casadi_path, render=False):
        self.render = render
        self.t = 0

        # Load the urdf model
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_model_path
        )
        print("model name: " + model.name)

        # Create data required by the algorithms
        data, collision_data, visual_data = pin.createDatas(
            model, collision_model, visual_model
        )

        # Sample a random configuration
        q = pin.randomConfiguration(model)
        print(f"q: {q.T}")

        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        frame_name = "end_effector"
        T_pin = data.oMf[model.getFrameId(frame_name)]
        print(f"T_pin: {T_pin}")

        self.viz = PMV(model, collision_model, visual_model, collision_data=collision_data, visual_data=visual_data)
        self.viz.initViewer(open=False)

        self.viz.loadViewerModel()
        q_init = np.array([0, 0])
        self.viz.display(q_init)

        self.f_state_transition = FurutaPendulumSimulator.create_f_state_transition(forward_dynamics_casadi_path) 

        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.x_hist = [self.x]

    @staticmethod
    def integrate_RK4(x_expr, u_expr, xdot_expr, dt, N_steps=1):

        h = dt/N_steps

        x_end = x_expr

        xdot_fun = ca.Function('xdot', [x_expr, u_expr], [xdot_expr])

        for _ in range(N_steps):

            k_1 = xdot_fun(x_end, u_expr)
            k_2 = xdot_fun(x_end + 0.5 * h * k_1, u_expr)
            k_3 = xdot_fun(x_end + 0.5 * h * k_2, u_expr)
            k_4 = xdot_fun(x_end + k_3 * h, u_expr)

            x_end = x_end + (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

        F_expr = x_end

        return F_expr
    @staticmethod
    def create_f_state_transition(forward_dynamics_casadi_path):
        # Create state transition function from forward dynamics
        q_sym = ca.MX.sym("q", 2)
        dq_sym = ca.MX.sym("dq", 2)
        x_sym = ca.vertcat(q_sym, dq_sym)
        u_sym = ca.MX.sym("u")

        forward_dynamics = ca.Function.load(forward_dynamics_casadi_path)
        ddq_sym = forward_dynamics(q_sym, dq_sym, ca.vertcat(u_sym,0))
        dx_sym = ca.vertcat(dq_sym, ddq_sym)
        dt_sym = ca.MX.sym("dt") 

        x_next_sym = FurutaPendulumSimulator.integrate_RK4(x_expr=x_sym, u_expr=u_sym, xdot_expr=dx_sym, dt=dt_sym, N_steps=1)
        f_state_transition = ca.Function("f_state_transition", [x_sym, u_sym, dt_sym], [x_next_sym])
        return f_state_transition  

    def step(self, u, dt):
        self.t += dt

        self.x = self.f_state_transition(self.x, u, dt).full().flatten()
        self.x_hist.append(self.x)

        if self.render:
            self.viz.display(self.x[:2])
            time.sleep(dt)

class FurutaPendulumTorqueEnv(gym.Env):
    def __init__(self, urdf_model_path, forward_dynamics_casadi_path, parameters_model, render=False, swingup = False):

        self.pendulum_sim = FurutaPendulumSimulator(urdf_model_path=urdf_model_path, 
                                                    forward_dynamics_casadi_path= forward_dynamics_casadi_path, 
                                                 render=render)
        
        if swingup:
            self.init_qpos = np.array([0.0, 0.0])
        else:
            # Start at upward position
            self.init_qpos = np.array([0.0, np.pi])

        self.init_qvel = np.array([0.0, 0.0])

        self.pendulum_sim.x = np.array([self.init_qpos[0], self.init_qpos[1], self.init_qvel[0], self.init_qvel[1]])
        self.qpos = np.array([
            self.pendulum_sim.x[0],
            self.pendulum_sim.x[1]
        ])
        self.qvel = np.array([
            self.pendulum_sim.x[2],
            self.pendulum_sim.x[3]
        ])

        self._max_velocity_joint0 = parameters_model["max_velocity_joint0"]
        self._max_velocity_joint1 = parameters_model["max_velocity_joint1"]
        self._max_angle_joint0 = parameters_model["max_angle_joint0"]
        self._max_angle_joint1 = parameters_model["max_angle_joint1"]
        self._max_torque_joint0 = parameters_model["max_torque_joint0"]


        self.J2 = parameters_model["J2"]
        self.J1 = parameters_model["J1"]
        self.l2 = parameters_model["L2"]

        self.m2 = parameters_model["m2"]
        self.m1 = parameters_model["m1"]

        self.g = 9.81

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)  # Action: motor torque [-1, 1]

        # Observation space: [sin(theta1), cos(theta1), sin(theta2), cos(theta2), dtheta1, dtheta2]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float64,
        )

        # Time parameters for the simulation
        self.dt = 0.01  # time step for simulation
        self.time_limit = 2000  # maximum number of steps per episode

        # Episode parameters
        self.time_step = 0
        self.done = False

        # if task includes swingup or not
        self.swingup = swingup

    
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]  

    def step(self, action):
        u = action[0]*self._max_torque_joint0 
        self.pendulum_sim.step(u, self.dt)
        self.qpos = np.array([
            self.pendulum_sim.x[0],
            self.pendulum_sim.x[1]
        ])
        self.qvel = np.array([
            self.pendulum_sim.x[2],
            self.pendulum_sim.x[3]
        ])
        obs = self._get_obs()
        reward = self.calculate_reward(obs, action)

        self.time_step += 1
        truncated = self.time_step >= self.time_limit

        if np.abs(self.qvel[0]) > self._max_velocity_joint0 or np.abs(self.qvel[1]) > self._max_velocity_joint1 or np.abs(self.qpos[0]) > self._max_angle_joint0: # if velocity is too high, terminate the episode and give a penalty
            terminated = True
            reward = -40
        elif not self.swingup and np.abs(self.qpos[1]-np.pi) > 45*np.pi/180: # if the pendulum is not in the upright position anymore, terminate the episode and give a penalty
            terminated = True
            reward = -40
        else:
            terminated = False

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        qpos = self.init_qpos
        qvel = self.init_qvel

        self.pendulum_sim.x = np.array([qpos[0], qpos[1], qvel[0], qvel[1]])
        self.qpos = np.array([qpos[0], qpos[1]])
        self.qvel = np.array([qvel[0], qvel[1]])

        self.time_step = 0

        return self._get_obs(), {}
    
    def _get_obs(self):
        # sin and cos instead of limit to get rid of discontinuities
        # scale angular velocities so that they won't dominate
        obs = np.array(
            [
                np.sin(self.qpos[0]),
                np.cos(self.qpos[0]),
                np.sin(self.qpos[1]),
                np.cos(self.qpos[1]),
                self.qvel[0] / self._max_velocity_joint0,
                self.qvel[1] / self._max_velocity_joint1,
                self.qpos[0] / self._max_angle_joint0,
                # self.qpos[1] / self._max_angle_joint1,
            ]
        )

        return obs

    def calculate_reward(self, obs: np.array, a: np.array):

        if not self.swingup:
            # Reward function parameters
            theta1_weight = 0.0
            theta2_weight = 10.0
            dtheta1_weight = 5.0
            dtheta2_weight = 5.0

            self._desired_obs_values = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
            self._obs_weights = [
                theta1_weight,
                theta1_weight,
                theta2_weight, 
                theta2_weight,
                dtheta1_weight,
                dtheta2_weight,
                theta1_weight,
            ]
            observation_reward = np.sum(
                [
                    -weight * np.power((desired_value - observation_value), 2)
                    for (observation_value, desired_value, weight) in zip(
                        obs, self._desired_obs_values, self._obs_weights
                    )
                ]
            )

            self._action_weight = 1.0
            action_reward = -self._action_weight * np.power(a[0], 2)

            reward = observation_reward + action_reward

            reward_normalized = reward / 4000

            return reward_normalized
        
        else:
            w_th1 = 0.0
            w_th2 = 10.0

            w_dth1 = 5.0
            w_dth2 = 5.0
            w_a = 0.0

            sin_th2, cos_th2 = obs[2], obs[3]
            dth1_norm, dth2_norm = obs[4], obs[5]

            E = 0.5 * self.J2 * (dth2_norm * self._max_velocity_joint1)**2 + self.m2 * self.g * self.l2 * (1 - cos_th2) + 0.5 * self.J1 * (dth1_norm * self._max_velocity_joint0)**2
            E_target = 2 * self.m2 * self.g * self.l2
            R_energy = - abs(E - E_target) / E_target

            R_stab = -( w_th2 * cos_th2 ) # you want cosine to be -1

            R_act = -w_a * a[0]**2

            k = 4
            alpha = 0.5 * (1 - np.tanh(k * cos_th2))

            reward = alpha * R_energy + (1 - alpha) * R_stab + R_act

            reward += 1.0 if abs(cos_th2 + 1) < 0.01 and abs(dth2_norm) < 0.05 else 0.0

            return reward
