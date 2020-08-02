#<!/bin/bash -x

prefix="SAC.MoPA.Ablation.piecewise0.7.ac_range1.0.reuse15.span1"
gpu=$1
seed=$2
algo='sac'
env="sawyer-push-obstacle-v2"
max_episode_step="250"
debug="False"
reward_type='sparse'
log_root_dir="/data/jun/projects/hrl-planner/logs"
# log_root_dir="./logs"
planner_integration="True"
reuse_data="True"
action_range="1.0"
omega='0.7'
stochastic_eval="True"
find_collision_free="True"
vis_replay="True"
plot_type='3d'
ac_space_type="piecewise"
use_smdp_update="True"
use_discount_meta="True"
step_size="0.02"
success_reward="150.0"
max_reuse_data='15'
reward_scale="0.2"
log_indiv_entropy="True"
evaluate_interval="10000"

# variants


python -m rl.main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --max_episode_step $max_episode_step \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --reward_type $reward_type \
    --planner_integration $planner_integration \
    --reuse_data $reuse_data \
    --action_range $action_range \
    --omega $omega
    --stochastic_eval $stochastic_eval \
    --find_collision_free $find_collision_free \
    --vis_replay $vis_replay \
    --plot_type $plot_type \
    --use_smdp_update $use_smdp_update \
    --ac_space_type $ac_space_type \
    --step_size $step_size \
    --success_reward $success_reward \
    --max_reuse_data $max_reuse_data \
    --reward_scale $reward_scale \
    --log_indiv_entropy $log_indiv_entropy \
    --evaluate_interval $evaluate_interval \
    --use_discount_meta $use_discount_meta \
