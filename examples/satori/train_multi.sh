set -x

PRETRAIN_MODEL=Satori-reasoning/Satori-SFT-7B
REWARD_MODEL=Satori-reasoning/Satori-RM-7B
PROMPT_PATH=Satori-reasoning/Satori_RL_data_with_RAE


ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --temperature 0.6 \
   --save_steps 20 \
   --ref_num_nodes 2 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 2 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 2 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 2 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain ${PRETRAIN_MODEL} \
   --reward_pretrain ${REWARD_MODEL} \
   --save_path models/satori/final \
   --ckpt_path models/satori/ckpt \
   --save_hf_ckpt \
   --max_ckpt_num 999 \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 1000000 \
   --max_epochs 1 \
   --bonus_scale 0.5 \
   --prompt_max_len 1024 \
   --generate_max_len 3072 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-7 \
   --critic_learning_rate 5e-6 \
   --init_kl_coef 0.0 \
   --prompt_data ${PROMPT_PATH} \
   --input_key query \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
