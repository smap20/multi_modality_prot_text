export CUDA_VISIBLE_DEVICES=3
python mask_prediction.py \
    --model_path /m-ent1/ent1/smap20/checkpoints/prot_pubmedbert \
    --mask_prot \
    --mask_len 3 \
    --batch_size 64 \
    --validation_file ../data/valid.txt \
    --amino ../data/amino.txt \

