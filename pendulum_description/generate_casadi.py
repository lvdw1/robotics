import numpy as np
import pinocchio as pin
from pinocchio.utils import *

import casadi as ca
from pinocchio.visualize import MeshcatVisualizer as PMV

urdf_model_path = "double_pendulum.urdf"

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

viz = PMV(model, collision_model, visual_model, collision_data=collision_data, visual_data=visual_data)
viz.initViewer(open=False)

viz.loadViewerModel()
# Display a robot configuration.
viz.display(q)
viz.displayVisuals(True)

q = pin.randomConfiguration(model)
q[0] = 0
q[1] = 0
q[2] = 0
print(f"q: {q.T}")
viz.display(q)
viz.displayVisuals(True)
viz.displayFrames(True)
viz.updateFrames()

# Jacobian: space Jacobian J_s and body Jacobian J_b
pin.computeJointJacobians(model, data, q)
J_s = pin.getJointJacobian(model, data, model.nq, pin.WORLD)
print(f"J_s: {J_s}")
J_b = pin.getFrameJacobian(model, data, model.getFrameId(frame_name), pin.LOCAL)
print(f"J_b: {J_b}")

# Check Jacobian using adjoint representation of T_pin (called action)
print(f"J_s check:{T_pin.action@J_b}")

# Forward dynamics (joint accelerations)
v = np.random.rand(model.nv, 1)  # random joint velocity
tau = np.random.rand(model.nv, 1)  # random joint torques
ddq = pin.aba(model, data, q, v, tau)
print(f"ddq: {ddq}")

# Inverse dynamics (joint torques)
tau = pin.rnea(model, data, q, v, ddq)
print(f"tau_est: {tau}")

import casadi as ca
from pinocchio import casadi as cpin

# Define casadi symbols for the joints
cq = ca.SX.sym('q', model.nq)
cdq = ca.SX.sym('dq', model.nq)
cddq = ca.SX.sym('ddq', model.nq)

# Define the equivalent casadi model and data
cmodel = cpin.Model(model)
cdata = cmodel.createData()

# Compute the forwards kinematics
cpin.framesForwardKinematics(cmodel, cdata, cq)

frame_name = "end_effector"

# Get the transformation matrix from the world to the frame
T_pin = cdata.oMf[model.getFrameId(frame_name)]

# Combine the rotation and translation into one matrix
T = ca.SX.zeros(4, 4)
T[:3, :3] = T_pin.rotation
T[:3, 3] = T_pin.translation
T[3, 3] = 1

# Simplify the matrix (be careful with this!)
T = ca.cse(ca.sparsify(T, 1e-10))
print(T)

# Define the casadi function
T_fk = ca.Function('T_fk', [cq], [T], ['q'], ['T_world_'+frame_name])


# ── 1. symbolic variables ──────────────────────────────────────────────────────
cq   = ca.SX.sym('q',  model.nq)   # joint positions
cdq  = ca.SX.sym('dq', model.nq)   # joint velocities
ctau = ca.SX.sym('tau', model.nv)  # joint torques
# ── 2. CasADi copy of the rigid-body model ─────────────────────────────────────
cmodel = cpin.Model(model)         # deep copy that is purely symbolic
cdata  = cmodel.createData()
# ── 3. forward dynamics (Articulated-Body Algorithm) ───────────────────────────
cddq = cpin.aba(cmodel, cdata, cq, cdq, ctau)   # symbolic joint accelerations
# ── 4. wrap it in a CasADi Function ────────────────────────────────────────────
forward_dynamics = ca.Function(
    'forward_dynamics',
    [cq, cdq, ctau],     # inputs
    [cddq],              # outputs
    ['q', 'dq', 'tau'],  # input names   (optional but handy)
    ['ddq']              # output names
)
# ── 5. write it to disk ────────────────────────────────────────────────────────
forward_dynamics.save('forward_dynamics2.casadi')
