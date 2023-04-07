export CUDA_VISIBLE_DEVICES=0
python sleep.py
python run_mlm_no_trainer.py \
    --train_file data/train_pure_text.txt \
    --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --output_dir /m-ent1/ent1/smap20/checkpoints/text_pubmedbert \
    --checkpointing_steps 20000 \
    --num_train_epochs 1 \
    --fp16 \
    --per_device_train_batch_size 16 \
    --preprocessing_num_workers 64 \