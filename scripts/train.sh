task_name='xarm_close_dex'
frames=801000
feature_dim=256
aux_lr=8e-5
use_wandb=True
save_snapshot=True
save_video=False
use_traj=False


CUDA_VISIBLE_DEVICES=0  python camera_train.py \
                            task@_global_=${task_name} \
                            seed=2 \
                            use_wandb=${use_wandb} \
                            num_train_frames=${frames} \
                            save_snapshot=${save_snapshot}  \
                            agent.aux_l2_coef=200 \
                            agent.aux_tcc_coef=0 \
                            agent.temp=0.1 \
                            agent.aux_coef=500 \
                            agent.aux_latency=200000 \
                            use_traj=${use_traj} \
                            use_depth=False \
                            wandb_group=$1