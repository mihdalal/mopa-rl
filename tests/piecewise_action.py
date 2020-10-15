import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config import argparser
import torch
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair, save_video
from util.gym import render_frame, observation_size, action_size
from config.motion_planner import add_arguments as planner_add_arguments
import torchvision
from rl.sac_agent import SACAgent
from rl.policies import get_actor_critic_by_name
import time
import timeit
import copy
np.set_printoptions(precision=3)

parser = argparser()
config, unparsed = parser.parse_known_args()
if 'pusher' in config.env:
    from config.pusher import add_arguments
    add_arguments(parser)
elif 'robosuite' in config.env:
    from config.robosuite import add_arguments
    add_arguments(parser)
elif 'sawyer' in config.env:
    from config.sawyer import add_arguments
    add_arguments(parser)
elif 'reacher' in config.env:
    from config.reacher import add_arguments
    add_arguments(parser)

planner_add_arguments(parser)
config, unparsed = parser.parse_known_args()
env = gym.make(config.env, **config.__dict__)
config._xml_path = env.xml_path
config.device = torch.device("cpu")
config.is_chef = False
config.planner_integration = True

ob_space = env.observation_space
ac_space = env.action_space
joint_space = env.joint_space

allowed_collsion_pairs = []
geom_ids = env.agent_geom_ids + env.static_geom_ids
if config.allow_manipulation_collision:
    for manipulation_geom_id in env.manipulation_geom_ids:
        for geom_id in geom_ids:
            allowed_collsion_pairs.append(make_ordered_pair(manipulation_geom_id, geom_id))

ignored_contact_geom_ids = []
ignored_contact_geom_ids.extend(allowed_collsion_pairs)
config.ignored_contact_geom_ids = ignored_contact_geom_ids

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]
config.passive_joint_idx = passive_joint_idx

actor, critic = get_actor_critic_by_name(config.policy)

# build up networks
non_limited_idx = np.where(env.sim.model.jnt_limited[:action_size(env.action_space)]==0)[0]
agent = SACAgent(
    config, ob_space, ac_space, actor, critic, non_limited_idx, env.ref_joint_pos_indexes, env.joint_space, env._is_jnt_limited, env.jnt_indices
)


x_values = np.linspace(0, 1.0, 100)
converted_list = []
for x in x_values:
    action = np.ones(3)  * x
    converted_list.append(agent.convert2planner_displacement(action, env._ac_scale)[0])

import matplotlib.pyplot as plt
plt.plot(x_values, converted_list)
plt.savefig("piecewise.png")

