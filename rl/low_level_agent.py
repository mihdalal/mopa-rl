import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

from rl.sac_agent import SACAgent
from rl.normalizer import Normalizer
from rl.mp_agent import MpAgent
from util.logger import logger
from util.pytorch import to_tensor, get_ckpt_path
from util.gym import action_size, observation_size
from util.mpi import mpi_average
from util.pytorch import optimizer_cuda, count_parameters, \
    compute_gradient_norm, compute_weight_norm, sync_networks, sync_grads, to_tensor, sync_avg_grads
from env.action_spec import ActionSpec
import math

from gym import spaces

from util.logger import logger
from rl.policies.mlp_actor_critic import MlpActor, MlpCritic

class LowLevelAgent(SACAgent):
    ''' Low level agent that includes skill sets for each agent, their
        execution procedure given observation and skill selections from
        meta-policy, and their training (for single-skill-per-agent cases
        only).
    '''

    def __init__(self, config, ob_space, ac_space, actor, critic, non_limited_idx, subgoal_space=None, subgoal_critic=None):
        self._non_limited_idx = non_limited_idx
        self._subgoal_space = subgoal_space
        super().__init__(config, ob_space, ac_space, actor, critic)
        self._log_alpha = [torch.zeros(1, requires_grad=True, device=config.device) for _ in range(len(config.primitive_skills))]
        self._alpha_optim = [optim.Adam([_log_alpha], lr=config.lr_actor) for _log_alpha in self._log_alpha]

    def _log_creation(self):
        if self._config.is_chef:
            logger.info('Creating a low-level agent')

    def _build_actor(self, actor):
        config = self._config

        # parse body parts and skills
        self._actors = []
        self._ob_norms = []
        self._planners = []

        # load networks
        #mp = MpAgent(config, ac_space, non_limited_idx)

        # Change here !!!!!!
        if config.primitive_skills:
            skills = config.primitive_skills
        else:
            skills = ['primitive']

        self._skills = skills
        planner_i = 0

        for skill in skills:
            if 'mp' in skill and self._subgoal_space is not None:
                skill_actor = actor(config, self._ob_space, self._subgoal_space, config.tanh_policy)
            else:
                skill_actor = actor(config, self._ob_space, self._ac_space, config.tanh_policy)
            skill_ob_norm = Normalizer(self._ob_space,
                                       default_clip_range=config.clip_range,
                                       clip_obs=config.clip_obs)

            if self._config.meta_update_target == 'HL':
                if "mp" not in skill:
                    path = os.path.join(config.primitive_dir, skill)
                    ckpt_path, ckpt_num = get_ckpt_path(path, None)
                    logger.warn('Load skill checkpoint (%s) from (%s)', skill, ckpt_path)
                    ckpt = torch.load(ckpt_path)

                    if type(ckpt['agent']['actor_state_dict']) == OrderedDict:
                        # backward compatibility to older checkpoints
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'])
                    else:
                        skill_actor.load_state_dict(ckpt['agent']['actor_state_dict'][0])
                    skill_ob_norm.load_state_dict(ckpt['agent']['ob_norm_state_dict'])

            if 'mp' in skill:
                ignored_contacts = config.ignored_contact_geom_ids[planner_i]
                passive_joint_idx = config.passive_joint_idx
                planner = MpAgent(config, self._ac_space, self._non_limited_idx, passive_joint_idx=passive_joint_idx, ignored_contacts=ignored_contacts)
                self._planners.append(planner)
                planner_i += 1
            else:
                self._planners.append(None)

            skill_actor.to(config.device)
            self._actors.append(skill_actor)
            self._ob_norms.append(skill_ob_norm)

    def _build_critic(self, critic):
        config = self._config
        if config.use_single_critic:
            super()._build_critic()
        else:
            self._critics1 = []
            self._critics2 = []
            self._critic1_targets = []
            self._critic2_targets = []
            for skill in config.primitive_skills:
                if 'mp' in skill:
                    self._critics1.append(critic(config, self._ob_space, self._subgoal_space))
                    self._critics2.append(critic(config, self._ob_space, self._subgoal_space))
                    self._critic1_targets.append(critic(config, self._ob_space, self._subgoal_space))
                    self._critic2_targets.append(critic(config, self._ob_space, self._subgoal_space))
                else:
                    self._critics1.append(critic(config, self._ob_space, self._ac_space))
                    self._critics2.append(critic(config, self._ob_space, self._ac_space))
                    self._critic1_targets.append(critic(config, self._ob_space, self._ac_space))
                    self._critic2_targets.append(critic(config, self._ob_space, self._ac_space))

            for i in range(len(self._critics1)):
                self._critic1_targets[i].load_state_dict(self._critics1[i].state_dict())
                self._critic2_targets[i].load_state_dict(self._critics2[i].state_dict())




    def plan(self, curr_qpos, target_qpos=None, meta_ac=None, ob=None, is_train=True, random_exploration=False, ref_joint_pos_indexes=None):
        assert len(self._planners) != 0, "No planner exists"

        activation = None
        if target_qpos is None:
            assert ob is not None and meta_ac is not None, "Invalid arguments"

            skill_idx = int(meta_ac['default'][0])
            assert self._planners[skill_idx] is not None

            assert "mp" in self.return_skill_type(meta_ac), "Skill is expected to be motion planner"
            if random_exploration:
                if self._subgoal_space is not None:
                    ac = self._subgoal_space.sample()
                else:
                    ac = self._ac_space.sample()
            else:
                ac, activation = self._actors[skill_idx].act(ob, is_train)
            target_qpos = curr_qpos.copy()
            if self._config.relative_goal:
                target_qpos[ref_joint_pos_indexes] += ac['default'][:len(ref_joint_pos_indexes)]
            else:
                target_qpos[ref_joint_pos_indexes] = ac['default'][:len(ref_joint_pos_indexes)]

            traj, success = self._planners[skill_idx].plan(curr_qpos, target_qpos)
            return traj, success, target_qpos, ac, activation
        else:
            traj, success = self._planners[0].plan(curr_qpos, target_qpos)
            return traj, success

    def act(self, ob, meta_ac, is_train=True, return_stds=False):
        if self._config.hrl:
            skill_idx = int(meta_ac['default'][0])
            if self._config.meta_update_target == 'HL':
                if return_stds:
                    ac, activation, stds = self._actors[skill_idx].act(ob, False, return_stds=return_stds)
                else:
                    ac, activation = self._actors[skill_idx].act(ob, False, return_stds=return_stds)
            else:
                if return_stds:
                    ac, activation, stds = self._actors[skill_idx].act(ob, is_train, return_stds=return_stds)
                else:
                    ac, activation = self._actors[skill_idx].act(ob, is_train, return_stds=return_stds)

        if return_stds:
            return ac, activation, stds
        else:
            return ac, activation

    def return_skill_type(self, meta_ac):
        skill_idx = int(meta_ac['default'][0])
        return self._skills[skill_idx]

    def act_log(self, ob, meta_ac=None):
        ''' Note: only usable for SAC agents '''
        skill_idx = int(meta_ac['default'][0])
        return self._actors[skill_idx].act_log(ob)

    def train(self):
        train_info = {}
        for i in range(self._config.num_batches):
            for skill_idx in range(len(self._config.primitive_skills)):
                if self._buffer._current_size[skill_idx] > self._config.batch_size * 10:
                    transitions = self._buffer.sample(self._config.batch_size, skill_idx)
                else:
                    transitions = self._buffer.create_empty_transition()
                info = self._update_network(transitions, i, skill_idx)
                train_info.update(info)
                self._soft_update_target_network(self._critic1_targets[skill_idx], self._critics1[skill_idx], self._config.polyak)
                self._soft_update_target_network(self._critic2_targets[skill_idx], self._critics2[skill_idx], self._config.polyak)

        train_info.update({
            'actor_grad_norm': np.mean([compute_gradient_norm(_actor) for _actor in self._actors]),
            'actor_weight_norm': np.mean([compute_weight_norm(_actor) for _actor in self._actors]),
            'critic1_grad_norm_{}'.format(self._config.primitive_skills[skill_idx]): compute_gradient_norm(self._critics1[skill_idx]),
            'critic2_grad_norm_{}'.format(self._config.primitive_skills[skill_idx]): compute_gradient_norm(self._critics2[skill_idx]),
            'critic1_weight_norm_{}'.format(self._config.primitive_skills[skill_idx]): compute_weight_norm(self._critics1[skill_idx]),
            'critic2_weight_norm_{}'.format(self._config.primitive_skills[skill_idx]): compute_weight_norm(self._critics2[skill_idx]),
        })
        # print(train_info)
        return train_info

    def sync_networks(self):
        if self._config.meta_update_target == 'LL' or \
           self._config.meta_update_target == 'both':
            super().sync_networks()
        else:
            pass

    def _zero_gradient(self, skill_idx):
        info = {}
        if self._config.use_single_critic:
            critic_idx = 0
        else:
            critic_idx = skill_idx
        self._actors[skill_idx].zero_grad()
        sync_avg_grads(self._actors[skill_idx])
        self._critics1[critic_idx].zero_grad()
        sync_avg_grads(self._critics1[critic_idx])
        self._critics2[critic_idx].zero_grad()
        sync_avg_grads(self._critics2[critic_idx])

        info['min_target_q'] = 0.
        info['target_q'] = 0.
        info['min_real1_q'] = 0.
        info['min_real2_q'] = 0.
        info['real1_q'] = 0.
        info['real2_q'] = 0.
        info['critic1_loss'] = 0.
        info['critic2_loss'] = 0.
        info['entropy_alpha'] = self._log_alpha[skill_idx].exp().cpu().item()
        info['entropy_loss'] = 0.
        info['actor_loss'] = 0.
        if len(self._actors) == 1:
            constructed_info = info
        else:
            constructed_info = {}
            for k, v in info.items():
                constructed_info['skill_{}/{}'.format(self._config.primitive_skills[skill_idx], k)] = v
        return mpi_average(constructed_info)

    def _update_network(self, transitions, step=0, skill_idx=None):
        info = {}

        # pre-process observations
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o, o_next = transitions['ob'], transitions['ob_next']

        if self._config.hrl:
            meta_ac = _to_tensor(transitions['meta_ac'])
        else:
            meta_ac = None
            skill_idx = 0

        if self._config.use_single_critic:
            critic_idx = 0
        else:
            critic_idx = skill_idx

        if len(o) == 0:
            return self._zero_gradient(skill_idx)


        bs = len(transitions['done'])
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions['ac'])
        done = _to_tensor(transitions['done']).reshape(bs, 1)
        rew = _to_tensor(transitions['rew']).reshape(bs, 1)

        # update alpha
        actions_real, log_pi = self.act_log(o, meta_ac=meta_ac)
        alpha_loss = -(self._log_alpha[skill_idx] * (log_pi + self._target_entropy[skill_idx]).detach()).mean()


        if self._config.use_automatic_entropy_tuning:
            self._alpha_optim[skill_idx].zero_grad()
            alpha_loss.backward()
            self._alpha_optim[skill_idx].step()
            self._log_alpha[skill_idx].data.clamp_(min=math.log(0.01), max=math.log(10))
            alpha = [_log_alpha.exp() for _log_alpha in self._log_alpha]
        else:
            alpha = [torch.ones(1).to(self._config.device) for _ in self._log_alpha]

        # the actor loss
        entropy_loss = (alpha[skill_idx] * log_pi).mean()
        actor_loss = -torch.min(self._critics1[critic_idx](o, actions_real),
                                self._critics2[critic_idx](o, actions_real)).mean()

        info['entropy_alpha'] = alpha[skill_idx].cpu().item()
        info['entropy_loss'] = entropy_loss.cpu().item()
        info['actor_loss'] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, log_pi_next = self.act_log(o_next, meta_ac=meta_ac)
            q_next_value1 = self._critic1_targets[critic_idx](o_next, actions_next)
            q_next_value2 = self._critic2_targets[critic_idx](o_next, actions_next)
            q_next_value = (torch.min(q_next_value1, q_next_value2) - alpha[skill_idx] * log_pi_next)
            target_q_value = rew * self._config.reward_scale + \
                (1 - done) * self._config.discount_factor * q_next_value
            target_q_value = target_q_value.detach()
            ## clip the q value
            # clip_return = 1 / (1 - self._config.discount_factor)
            # target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # the q loss
        real_q_value1 = self._critics1[critic_idx](o, ac)
        real_q_value2 = self._critics2[critic_idx](o, ac)
        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info['min_target_q'] = target_q_value.min().cpu().item()
        info['target_q'] = target_q_value.mean().cpu().item()
        info['min_real1_q'] = real_q_value1.min().cpu().item()
        info['min_real2_q'] = real_q_value2.min().cpu().item()
        info['real1_q'] = real_q_value1.mean().cpu().item()
        info['real2_q'] = real_q_value2.mean().cpu().item()
        info['critic1_loss'] = critic1_loss.cpu().item()
        info['critic2_loss'] = critic2_loss.cpu().item()

        # update the actor
        #for _actor_optim in self._actor_optims:
        self._actor_optims[skill_idx].zero_grad()
        actor_loss.backward()
        sync_avg_grads(self._actors[skill_idx])
        self._actor_optims[skill_idx].step()

        # update the critic
        self._critic1_optims[critic_idx].zero_grad()
        critic1_loss.backward()
        sync_avg_grads(self._critics1[critic_idx])
        self._critic1_optims[critic_idx].step()

        self._critic2_optims[critic_idx].zero_grad()
        critic2_loss.backward()
        sync_avg_grads(self._critics2[critic_idx])
        self._critic2_optims[critic_idx].step()

        # include info from policy
        if len(self._actors) == 1:
            constructed_info = info
        else:
            constructed_info = {}
            for k, v in info.items():
                constructed_info['skill_{}/{}'.format(self._config.primitive_skills[skill_idx], k)] = v
        return mpi_average(constructed_info)
