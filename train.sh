torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
main_train.py \
    --world_size 1 \
    --batch_size 16 \
    --data_path "/CASIA2.0" \
    --epochs 200 \
    --lr 2e-4 \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --pretrain_path "/uniformer_base_ls_in1k.pth" \
    --test_data_path "/CASIA1.0" \
    --warmup_epochs 4 \
    --output_dir ./output_dir_window_overall/ \
    --log_dir ./output_dir_window_overall/  \
    --accum_iter 16 \
    --seed 42 \
    --test_period 4 \
    --num_workers 8 \
    2> train_error.log 1>train_log.log