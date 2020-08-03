python -m rl.main --log_root_dir /data/jun/projects/hrl-planner/logs --wandb True --prefix BASELINE.v26 --max_global_step 1500000 --env sawyer-assembly-v0 --gpu 0 --max_episode_step 250 --evaluate_interval 10000 --buffer_size 1000000 --num_batches 1 --debug False --rollout_length 10000 --batch_size 256 --rl_activation relu --algo sac --seed 1237 --reward_type sparse --comment Baseline --start_steps 10000 --log_interval 1000 --alpha 1.0 --vis_replay True --task_level easy --plot_type 3d --success_reward 150. --reward_scale 10. --use_ik_target False --ckpt_interval 100000 --ik_target grip_site --action_range 0.001 --is_train False --vis_info False --camera_name zoomview --date 07.25
