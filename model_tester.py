import torch
from ppo_continuous_action import Agent, make_env
import gymnasium as gym
from furuta_torque_env import FurutaPendulumTorqueEnv
import yaml 
import os

with open('pendulum_description/simulation_pendulum.yaml', 'r') as file:
        config = yaml.safe_load(file)

parameters_model = config["parameters_model"]
urdf_path = os.path.join("pendulum_description", config["urdf_filename"])
forward_dynamics_casadi_path = os.path.join("pendulum_description", config["forward_dynamics_casadi_filename"])
gym.register(
    id="FurutaPendulumTorque-v0",
    entry_point=FurutaPendulumTorqueEnv,
)

# Create an instance of the FurutaPendulumTorqueEnv environment
env = gym.make("FurutaPendulumTorque-v0",
               urdf_model_path=urdf_path, forward_dynamics_casadi_path=forward_dynamics_casadi_path, parameters_model=parameters_model, render=True, swingup=True)

saved_model_path = '/Users/louis/MAI/semester2/Robotics/project/furuta_pendulum_tensorboard/furutaTorque__ppo_continuous_action__42__1745696792/ppo_continuous_action.cleanrl_model' 

swingup = True

envs = gym.vector.SyncVectorEnv(
        [make_env(urdf_path=urdf_path, 
                  parameters_model=parameters_model, 
                  forward_dynamics_casadi_path=forward_dynamics_casadi_path,
                  render=True, swingup=swingup) for _ in range(1)]
    )

model = Agent(envs=envs)

model.load_state_dict(torch.load(saved_model_path, map_location="cpu"))
model.eval()

observation, info = envs.reset()
terminated = False
truncated = False
print(f"Initial Observation: {observation}")
# Apply a sequence of actions
try:
    while True:
        action, _, _, _ = model.get_action_and_value(torch.Tensor(observation), deterministic=True)  # Use the trained model to predict the action
        action = action.cpu().detach().numpy()
        observation, reward, terminated, truncated, info = envs.step(action)
        print(f"Observation: {observation}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            observation, info = envs.reset()
except KeyboardInterrupt:
    envs.close()
