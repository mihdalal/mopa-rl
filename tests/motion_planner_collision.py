import os, sys
import numpy as np
import shutil
from collections import OrderedDict
import gym
import env
from config.pusher import add_arguments
from config import argparser
from rl.planner_agent import PlannerAgent
from util.misc import make_ordered_pair
from config.motion_planner import add_arguments as planner_add_arguments


def render_frame(env, step, info={}):
    color = (200, 200, 200)
    text = "Step: {}".format(step)
    frame = env.render('rgb_array') * 255.0
    fheight, fwidth = frame.shape[:2]
    frame = np.concatenate([frame, np.zeros((fheight, fwidth, 3))], 0)

    if record_caption:
        font_size = 0.4
        thickness = 1
        offset = 12
        x, y = 5, fheight+10
        cv2.putText(frame, text,
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, (255, 255, 0), thickness, cv2.LINE_AA)

        for i, k in enumerate(info.keys()):
            v = info[k]
            key_text = '{}: '.format(k)
            (key_width, _), _ = cv2.getTextSize(key_text, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_size, thickness)
            cv2.putText(frame, key_text,
                        (x, y+offset*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (66, 133, 244), thickness, cv2.LINE_AA)
            cv2.putText(frame, str(v),
                        (x + key_width, y+offset*(i+2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame





parser = argparser()
add_arguments(parser)
planner_add_arguments(parser)
args, unparsed = parser.parse_known_args()

args.env = 'simple-pusher-obstacle-v0'
env = gym.make(args.env, **args.__dict__)
mp_env = gym.make(args.env, **args.__dict__)
args._xml_path = env.xml_path
args.planner_type="sst"
args.planner_objective="state_const_integral"
args.range=0.1
args.threshold=0.1
args.timelimit=0.01

ignored_contacts = []
# Allow collision with manipulatable object
geom_ids = [env.sim.model.geom_name2id(name) for name in env.agent_geoms]
manipulation_geom_ids = [env.sim.model.geom_name2id(name) for name in env.manpulation_geom]
for manipulation_geom_id in manipulation_geom_ids:
    for geom_id in geom_ids:
        ignored_contacts.append(make_ordered_pair(manipulation_geom_id, geom_id))

passive_joint_idx = list(range(len(env.sim.data.qpos)))
[passive_joint_idx.remove(idx) for idx in env.ref_joint_pos_indexes]

non_limited_idx = np.where(env._is_jnt_limited==0)[0]
planner = PlannerAgent(args, env.action_space, non_limited_idx, passive_joint_idx, ignored_contacts)


N = 5
save_video = False
frames = []
for _ in range(N):
    done = False
    ob = env.reset()
    step = 0
    if save_video:
        frames.add([render_frame(env, step)])
    else:
        env.render('human')

    while not done:
        current_qpos = env.sim.data.qpos.copy()
        target_qpos = current_qpos.copy()
        target_qpos[env.ref_joint_pos_indexes] += np.random.uniform(low=-1, high=1, size=len(env.ref_joint_pos_indexes))
        traj, success = planner.plan(current_qpos, target_qpos)
        mp_env.set_state(target_qpos, env.sim.data.qvel.ravel().copy())
        xpos = OrderedDict()
        xquat = OrderedDict()

        for i in range(len(mp_env.ref_joint_pos_indexes)):
            name = 'body'+str(i)
            body_idx = mp_env.sim.model.body_name2id(name)
            key = name+'-goal'
            env._set_pos(key, mp_env.sim.data.body_xpos[body_idx].copy())
            env._set_quat(key, mp_env.sim.data.body_xquat[body_idx].copy())
            color = env._get_color(key)
            color[-1] = 0.3
            env._set_color(key, color)

        if success:
            for j, next_qpos in enumerate(traj):
                action = env.form_action(next_qpos)
                ob, reward, done, info = env.step(action, is_planner=True)

                mp_env.set_state(next_qpos, env.sim.data.qvel.copy())
                for i in range(len(mp_env.ref_joint_pos_indexes)):
                    name = 'body'+str(i)
                    body_idx = mp_env.sim.model.body_name2id(name)
                    key = name+'-dummy'
                    env._set_pos(key, mp_env.sim.data.body_xpos[body_idx].copy())
                    env._set_quat(key, mp_env.sim.data.body_xquat[body_idx].copy())
                    color = env._get_color(key)
                    color[-1] = 0.3
                    env._set_color(key, color)

                step += 1
                if save_video:
                    frames[i].append(render_frame(env, step))
                else:
                    env.render('human')
        else:
            import pdb; pdb.set_trace()


if save_video:
    prefix_path = './tmp/motion_planning_test/'
    for i, episode_frames in enumerate(frames):
        fpath = os.path.join(prefix_path, 'test_trial_{}'.format(i))
        save_video(fpath, episode_frames, fps=5)
