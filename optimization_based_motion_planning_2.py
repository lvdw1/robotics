import time
import numpy as np
from furuta_torque_env_2 import FurutaPendulumTorqueEnv
from rockit import MultipleShooting, Ocp
import casadi as ca
import torch
from ppo_continuous_action import Agent, make_env
import gymnasium as gym

class OptiPlannerBase:
    def __init__(self, urdf_model_path, forward_dynamics_casadi_path, parameters_model):
        self.f_state_transition = FurutaPendulumTorqueEnv(  # dummy env to load dynamics
            urdf_model_path=urdf_model_path,
            forward_dynamics_casadi_path=forward_dynamics_casadi_path,
            parameters_model=parameters_model,
            render=True
        ).pendulum_sim.f_state_transition
        self.parameters_model = parameters_model
        self.urdf_model_path = urdf_model_path
        self.forward_dynamics_casadi_path = forward_dynamics_casadi_path

        time.sleep(10)

        self.ocp = None
        self.x = None
        self.u = None

        self.setup()

    def setup(self):
        """
        Setup the optimization problem.
        Make sure to set self.ocp, self.x, self.u
        """
        raise NotImplementedError

    def solve(self, N):
        """
        Solve the optimization problem.
        """
        method = MultipleShooting()
        self.ocp.method(method)
        self.sol = self.ocp.solve()

    def get_trajectory(self):
        """
        Return the state trajectory (unified state z) sampled on the control grid.
        """
        # Sampling on the 'control' grid: shape (state_dimension, N_time_steps)
        _, traj = self.sol.sample(self.x, grid='control')
        return traj

    def get_control_trajectory(self):
        """
        Return the control trajectory (torque inputs) sampled on the control grid.
        """
        # Sampling the control variable; shape (control_dimension, N_time_steps)
        _, u_traj = self.sol.sample(self.u, grid='control')
        return u_traj

class OptiPlannerSwingUp(OptiPlannerBase):
    def __init__(self, urdf_model_path, forward_dynamics_casadi_path, parameters_model):
        # Initial condition: [q0, q1, dq0, dq1]
        self.x_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Goal condition: [q0, q1, dq0, dq1]
        self.x_goal = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
        self.x_min = [ -parameters_model["max_angle_joint0"], -parameters_model["max_angle_joint1"], -parameters_model["max_angle_joint2"]]
        self.x_max = [  parameters_model["max_angle_joint0"],
                       parameters_model["max_angle_joint1"], parameters_model["max_angle_joint2"] ]
        self.v_max = [ parameters_model["max_velocity_joint0"],
                       parameters_model["max_velocity_joint1"], parameters_model["max_velocity_joint2"] ]
        self.u_max = 1

        super().__init__(urdf_model_path, forward_dynamics_casadi_path, parameters_model)

    def setup(self):
        # Create the OCP with initial time and total duration
        ocp = Ocp(t0=0, T=3)
    
        u = ocp.control(1)  # Torque input
    
        # Set a small integration time step (should be aligned with your simulation)
        dt_val = 0.01
    
        # Define the position state (angles)
        q = ocp.state(3)
        # Define the velocity state (angular velocities)
        dq = ocp.state(3)
        # Define your unified state as the concatenation of q and dq (for constraints and objective if needed)
        z = ca.vertcat(q, dq)

        actual_torque = u * self.parameters_model["max_torque_joint0"]
        z_next = self.f_state_transition(z, actual_torque, dt_val)
        # Approximate the derivative using finite difference from the one-step integration

        # Set the derivative for the velocity state:
        ocp.set_next(z, z_next)

        # Initial and final state constraints on the full state vector
        ocp.subject_to(ocp.at_t0(z) == self.x_start)
        ocp.subject_to(ocp.at_tf(z[1:]) == self.x_goal[1:])
    
        # Enforce joint angle constraints (on the first two dimensions of z)
        ocp.subject_to(z[0] >= self.x_min[0])
        ocp.subject_to(z[0] <= self.x_max[0])
        ocp.subject_to(z[1] >= self.x_min[1])
        ocp.subject_to(z[1] <= self.x_max[1])
        ocp.subject_to(z[2] >= self.x_min[2])
        ocp.subject_to(z[2] <= self.x_max[2])
    
        # Enforce velocity constraints (on the last two dimensions of z)
        ocp.subject_to(z[3] >= -self.v_max[0])
        ocp.subject_to(z[3] <= self.v_max[0])
        ocp.subject_to(z[4] >= -self.v_max[1])
        ocp.subject_to(z[4] <= self.v_max[1])
        ocp.subject_to(z[5] >= -self.v_max[2])
        ocp.subject_to(z[5] <= self.v_max[2])
    
        # Torque constraints
        ocp.subject_to(u >= -self.u_max)
        ocp.subject_to(u <= self.u_max)
    
        # Objective: Minimize control effort (using u^2 integrated over time)
        ocp.add_objective(ocp.sum(ca.sumsqr(u)))

        # Solver options
        options = {"expand": True, 'print_time': True, 'ipopt.print_level': 4}
        ocp.solver('ipopt', options)
    
        # Save variables for later use
        self.ocp = ocp
        self.x = z
        self.u = u

    def play_trajectory(self, dt_val=0.01):
        """
        Play the planned trajectory in the environment by applying the torque inputs.
        This method saves both the state and control (torque) trajectories.
        """
        # Instantiate the environment with rendering enabled and swingup mode as desired
        env = FurutaPendulumTorqueEnv(
            urdf_model_path=self.urdf_model_path,
            forward_dynamics_casadi_path=self.forward_dynamics_casadi_path,
            parameters_model=self.parameters_model,
            render=True,
            swingup=True  # or False if desired
        )

        # Reset the environment to its initial state
        env.reset()

        time.sleep(10)

        # Obtain the planned state and control trajectories from the OCP solution.
        state_traj = self.get_trajectory()       # Expected shape: (501, 4)
        control_traj = self.get_control_trajectory()  # Expected shape: (501,) or (1, 501)

        print("State trajectory shape:", state_traj.shape)
        print("Control trajectory shape:", control_traj.shape)

        # Save the trajectories for future use
        np.save("state_trajectory.npy", state_traj)
        np.save("u_trajectory.npy", control_traj)

        print("Playing planned trajectory using gym env step()...")

        num_steps = control_traj.shape[0]
        for i in range(num_steps):
            print(1)
            # Create a one-element action array
            action = np.array([control_traj[i]])
            # Step the simulation with the torque input (env.step expects an action)
            env.step(action)
            print("Action:", action, "State:", env.pendulum_sim.x)
            # Update visualization (showing first two joint angles)
            if env.pendulum_sim.render:
                env.pendulum_sim.viz.display(env.pendulum_sim.x[:2])
            time.sleep(dt_val)

        print("Finished playing trajectory.")

    def simulate_model_based_and_PPO(self, dt_val=0.01):
        """
        Simulate the model-based and PPO trajectories.
        This method is a placeholder for future implementation.
        """


        saved_model_path = 'furuta_pendulum_tensorboard/furutaTorque__ppo_continuous_action__42__1744890705/ppo_continuous_action.cleanrl_model' # no swingup

        # Set to False if task includes swingup
        swingup = True

        envs = gym.vector.SyncVectorEnv(
                [make_env(urdf_path=urdf_model_path, 
                        parameters_model=parameters_model, 
                        forward_dynamics_casadi_path=forward_dynamics_casadi_path,
                        render=True, swingup=swingup) for _ in range(1)]
            )

        model = Agent(envs=envs)

        model.load_state_dict(torch.load(saved_model_path, map_location="cpu"))
        
        envs.reset()

        time.sleep(10)

        # Obtain the planned state and control trajectories from the OCP solution.
        state_traj = self.get_trajectory()       # Expected shape: (501, 4)
        control_traj = self.get_control_trajectory()  # Expected shape: (501,) or (1, 501)

        print("State trajectory shape:", state_traj.shape)
        print("Control trajectory shape:", control_traj.shape)

        # Save the trajectories for future use
        np.save("state_trajectory.npy", state_traj)
        np.save("u_trajectory.npy", control_traj)

        print("Playing planned trajectory using gym env step()...")

        num_steps = control_traj.shape[0]
        for i in range(num_steps):
            action = np.array([control_traj[i]])
            observation, reward, terminated, truncated, info = envs.step(action)
            time.sleep(dt_val)
        

        while not terminated and not truncated:
            action, _, _, _ = model.get_action_and_value(torch.Tensor(observation), deterministic=True)  # Use the trained model to predict the action
            action = action.cpu().detach().numpy()
            observation, reward, terminated, truncated, info = envs.step(action)
        envs.close()


if __name__ == "__main__":
    # Example usage:
    # (Assumes that 'pendulum_description/simulation_pendulum.yaml' contains the required configuration.)
    import yaml, os
    with open('pendulum_description/simulation_double_pendulum.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    parameters_model = config["parameters_model"]
    urdf_model_path = os.path.join("pendulum_description", config["urdf_filename"])
    forward_dynamics_casadi_path = os.path.join("pendulum_description", config["forward_dynamics_casadi_filename"])
    
    planner = OptiPlannerSwingUp(urdf_model_path, forward_dynamics_casadi_path, parameters_model)
    planner.solve(N=300)
    # planner.play_trajectory(dt_val=0.01)

    planner.simulate_model_based_and_PPO(dt_val=0.01)


