# Sementic Search with Transformers and PEFT
This repository houses the code for the sementic search
# Dependencies
Ensure you have the following dependencies installed before running the code:

- Transformers
- PEFT
- Accelerate
- HNSW

# Commands

- GPU config
    - accelerate lauch 

- Training
    - python contrastive_train/peft_lora_constrastive_learning.py  --dataset_path "./dataset" --model_name_or_path "BAAI/bge-large-en" --output_dir "./sementic_search_outs" --use_peft  --with_tracking --report_to all

- Evaluation
    - python contrastive_train/test_with_hnsw.py --dataset_path "./dataset" --model_name_or_path "BAAI/bge-large-en" --peft_model_path "./sementic_search_outs"