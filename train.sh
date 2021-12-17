python scripts/train.py \
--encoder_h_dim_d 48 \
--neighborhood_size 2.0 \
--clipping_threshold_d 0 \
--clipping_threshold_g 2.0 \
--delim 'tab' \
--print_every 100 \
--pred_len 8 \
--loader_num_workers 0 \
--d_steps 1 \
--batch_size 64 \
--num_epochs 200 \
--num_layers 1 \
--best_k 20 \
--obs_len 8 \
--skip 1 \
--g_steps 1 \
--g_learning_rate 0.0001 \
--l2_loss_weight 1.0 \
--grid_size 8 \
--bottleneck_dim 8 \
--checkpoint_name 'checkpoint' \
--gpu_num '0' \
--restore_from_checkpoint 1 \
--dropout 0.0 \
--checkpoint_every 300 \
--noise_mix_type 'global' \
--decoder_h_dim_g 32 \
--pooling_type 'pool_net' \
--use_gpu 0 \
--num_iterations 9012 \
--noise_type 'gaussian' \
--encoder_h_dim_g 32 \
--timing 0 \
--d_learning_rate 0.001 \
--checkpoint_start_from None \
--embedding_dim 16 \
--mlp_dim 64 \
--num_samples_check 5000 \
--d_type 'global' \
--noise_dim 8 \
--dataset_name 'eth' \