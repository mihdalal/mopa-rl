import numpy as np
import gym
from tqdm import tqdm
from gym import spaces
import sys
import env
import cv2
from env.inverse_kinematics import qpos_from_site_pose_sampling, qpos_from_site_pose
from util.env import joint_convert, mat2quat, quat_mul, rotation_matrix, quat2mat
from config import sawyer
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from config.default_configs import LIFT_CONFIG
import mujoco_py

def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.
    Args:
        q (np.array): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")

def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.
    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.
    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat

def mat2pose(hmat):
    """
    Converts a homogeneous 4x4 matrix into pose.
    Args:
        hmat (np.array): a 4x4 homogeneous matrix
    Returns:
        2-tuple:
            - (np.array) (x,y,z) position array in cartesian coordinates
            - (np.array) (x,y,z,w) orientation array in quaternion form
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn

def get_site_pose(env, name):
    xpos = env.sim.data.get_site_xpos(name)[: len(env.min_world_size)].copy()
    # manually load xquat 
    model = env.sim.model
    xquat = mat2quat(env.sim.data.get_site_xmat(name).copy())
    # im assumign this is some type of conversion they already to from 
    # xyzw to wxyz
    xquat = convert_quat(xquat)
    return xpos, xquat

def get_object_pose(env, name):
    # object should be a free joint
    # note this is returned in xyzw assuming originally we had wxyz
    # from mujoco 
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    xpos = env.sim.data.qpos[start:start+3].copy()
    xquat = env.sim.data.qpos[start+3:start+7].copy()
    xquat = convert_quat(xquat)
    return np.concatenate((xpos, xquat))

def set_object_pose(env, name, new_xpos, new_xquat):
    start = env.sim.model.body_jntadr[env.sim.model.body_name2id(name)]
    # convert xquat
    new_xquat = convert_quat(new_xquat, to='wxyz')
    env.sim.data.qpos[start:start+3] = new_xpos
    env.sim.data.qpos[start+3:start+7] = new_xquat

def set_robot_based_on_ee_pos(
    env,
    ac,
    ik_env,
    qpos,
    qvel, 
    is_grasped,
    config,
):
    """
    - format target xyz by clipping 
    - also clip quaternion 
    - ik_env set state from curr qpos and qvel from env
    - call qpos_from_site_pose with desired stuff (maybe can set tol lower, is 1e-2 for now)
    - set state of env to that of ik env?? check to see if this works ok
    - return observation by getting them from the environment
    NOTES:
        - note ac should be something of the form ["default", "quat"] as an ordered dict
        - action is also a delta, not a desired position
        - min world size should be in 3d, only considering 3d tasks for now
        - replace everything with step and see how it works

    replace with end effector 
    how mp_env does it 
    - get position of end effector
    - perform everything the same with end effector
    what we should do
    - maybe try replacing with gripper info? since gripper should have same information
    - try replacing with actual end effector information
    - see what is the issue with grip site and debug from there
    """
    # keep track of gripper pos, etc
    gripper_qpos = env.sim.data.qpos[env.ref_gripper_joint_pos_indexes].copy()
    gripper_qvel = env.sim.data.qvel[env.ref_gripper_joint_pos_indexes].copy()
    object_pose = np.concatenate([
        env.sim.data.get_body_xpos('cube'),
        convert_quat(env.sim.data.get_body_xquat('cube'))
    ])
    old_eef_xpos, old_eef_xquat = get_site_pose(env, config['ik_target'])
    object_pose = get_object_pose(env, "cube").copy()
    target_cart = np.clip(
        env.sim.data.get_site_xpos(config["ik_target"])[: len(env.min_world_size)]
        + config["action_range"] * ac["default"],
        env.min_world_size,
        env.max_world_size,
    )
    if "quat" in ac.keys():
        target_quat = mat2quat(env.sim.data.get_site_xmat(config["ik_target"]))
        target_quat = target_quat[[3, 0, 1, 1]]
        target_quat = quat_mul(
            target_quat,
            (ac["quat"] / np.linalg.norm(ac["quat"])).astype(np.float64),
        )
    else:
        target_quat = None
    ik_env.set_state(env.sim.data.qpos.copy(), env.data.qvel.copy())
    result = qpos_from_site_pose(
        ik_env,
        config["ik_target"],
        target_pos=target_cart,
        target_quat=target_quat,
        rot_weight=2.0,
        joint_names=env.robot_joints,
        max_steps=100,
        tol=1e-2,
    )
    # set state here 
    env.set_state(ik_env.sim.data.qpos.copy(), ik_env.sim.data.qvel.copy())
    is_grasped = True
    if is_grasped:
        """
        - keep track of gripper velocity and gripper qpos - use ref gripper pos indexes
        - set current environment gripper qpos and qvel to old gripper qpos and qvel
        - end effector old pose, end effector new pose, compute transform using pose2mat
        - compute new transform
        - get new object pose by applying transform to object 
        """
        env.sim.data.qpos[env.ref_gripper_joint_pos_indexes] = gripper_qpos 
        env.sim.data.qvel[env.ref_gripper_joint_pos_indexes] = gripper_qvel

        # compute transform between new and old 
        ee_old_mat = pose2mat((old_eef_xpos, old_eef_xquat))
        new_eef_xpos, new_eef_xquat = get_site_pose(env, config['ik_target'])
        # new_eef_xpos = env.sim.data.get_site_xpos(config["ik_target"])[: len(env.min_world_size)].copy()
        # new_eef_xquat = convert_quat(env.sim.data.get_site_xquat(config["ik_target"])[: len(env.min_world_size)].copy())
        ee_new_mat = pose2mat((new_eef_xpos, new_eef_xquat))
        transform = ee_new_mat @ np.linalg.inv(ee_old_mat)
        
        # get new object pose
        new_object_pose = mat2pose(
            np.dot(transform, pose2mat((object_pose[:3], object_pose[3:])))
        )
        set_object_pose(env, "cube", new_object_pose[0], new_object_pose[1])
        env.sim.forward()

    return result.success, result.err_norm

def get_video(frames):
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

def main():
    """
    Create env, ik_env, look through config dict to get environments
    """
    np.random.seed(0)
    sawyer_config = sawyer.get_default_config()
    env = gym.make(**LIFT_CONFIG)
    ik_env = gym.make(**LIFT_CONFIG)
    env.reset()
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig("test.png")
    # get second image after teleporting 
    gripper_pos = env.sim.data.get_site_xpos(LIFT_CONFIG["ik_target"])[: len(env.min_world_size)]
    old_gripper_pos = gripper_pos.copy()
    cube_pos = env.sim.data.get_body_xpos("cube")
    print(f"Distance between gripper and cube: {np.linalg.norm(gripper_pos - cube_pos)}")
    print(f"Desired position: {cube_pos + np.array([0.0, 0.0, 0.01])}")
    print(f"Current position: {gripper_pos}")
    # desired action 
    ac = OrderedDict()
    ac['default'] = cube_pos - gripper_pos 
    # run teleporting 
    set_robot_based_on_ee_pos(
        env,
        ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        False,
        LIFT_CONFIG)
    # env.sim.data.qpos[env.ref_joint_pos_indexes] += converted_ac['default']
    # env.sim.forward()
    #print(f"Position at end ")
    #gripper_pos = env.sim.data.get_site_xpos(LIFT_CONFIG["ik_target"])[: len(env.min_world_size)]
    #print(np.linalg.norm(gripper_pos - cube_pos))
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig("test2.png")
    gripper_pos = env.sim.data.get_site_xpos(LIFT_CONFIG["ik_target"])[: len(env.min_world_size)]
    print(f"New gripper pos: {gripper_pos}")
    # try gripping 
    ac = OrderedDict()
    ac['default'] = np.array([0., 0., 0.25])
    set_robot_based_on_ee_pos(
        env,
        ac,
        ik_env,
        env.sim.data.qpos.copy(),
        env.sim.data.qvel.copy(), 
        True,
        LIFT_CONFIG)
    frame = env.render("rgb_array")
    plt.imshow(frame)
    plt.savefig("test3.png")
    print(f"DONE")

if __name__ == "__main__":
    main()