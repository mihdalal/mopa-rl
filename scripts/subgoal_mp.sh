python -m rl.main --log_root_dir ./logs --wandb True --prefix hl.sac.prm.kinematics.v4 --max_global_step 6000000 --hrl True --ll_type mp --planner_type prm_star --planner_objective state_const_integral --range 1.0 --threshold 0.2 --timelimit 1.0 --env reacher-obstacle-v0  --hl_type subgoal --gpu 2 --rl_hid_size 512 --meta_update_target both --hrl_network_to_update HL --max_mp_steps 30 --kinematics True --construct_time 300
