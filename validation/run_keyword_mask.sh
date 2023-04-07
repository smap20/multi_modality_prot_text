export CUDA_VISIBLE_DEVICES=2
python mask_prediction.py \
    --model_path /m-ent1/ent1/smap20/checkpoints/prot_pubmedbert \
    --mask_keyword \
    --batch_size 1 \
    --k 10 \
    --keyword_file ../data/treeNumberdict_disease.txt \
    --validation_file ../data/valid_Disease_MeSH_filt.txt \
