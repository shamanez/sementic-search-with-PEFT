import argparse
from tqdm import tqdm
import numpy as np

import datasets

import evaluate
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, default_data_collator

from accelerate.logging import get_logger
from peft import PeftModel

from base_model import AutoModelForSentenceEmbedding
from test_utils import preprocess_category_function, preprocess_title_function, construct_search_index, get_query_embeddings, get_nearest_neighbours, calculate_precision_recall

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Testing a PEFT model for Sematic Search task")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset path in the local dir")
    parser.add_argument("--cat_column_name", type=str, default="cat", help="name of the category col")
    parser.add_argument("--title_column_name", type=str, default="x_title", help="name of the title col")
    parser.add_argument("--embed_dim", type=int, default=1024, help="dimension of the model embedding")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        help="Path to the finetunned peft layers",
        required=True,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K retrieval",
    )
    args = parser.parse_args()

    return args


def main():
    
    args = parse_args()
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
    # load indexed categories
    cat_dataset = datasets.load_dataset("csv", data_files={"cat_index": f"{args.dataset_path}/cat_index.csv"})['cat_index']
    
    
    #############
    # load the test dataset
    test_title_dataset = datasets.load_dataset("csv", data_files={"test": f"{args.dataset_path}/test.csv"})['test']
        
    ############
    
    ids_to_cat_dict = {i: p for i, p in zip(cat_dataset["index"], cat_dataset["cat"])}
    
    
    # tokenization
    tokenized_cats = cat_dataset.map(
    preprocess_category_function,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer, "max_len": args.max_length},
    remove_columns=cat_dataset.column_names,
    desc="Running tokenizer on the category dataset",
    )
    
    tokenized_test_titles = test_title_dataset.map(
    preprocess_title_function,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer, "max_len": args.max_length},
    desc="Running tokenizer on the test title dataset",
    )
        
    cat_dataloader = DataLoader(
        tokenized_cats,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.test_batch_size,
        pin_memory=True,
    )
    
        
    # base model
    model = AutoModelForSentenceEmbedding(args.model_name_or_path, tokenizer)

    # peft config and wrapping
    model = PeftModel.from_pretrained(model, args.peft_model_path)
            
    model.to(args.device)
    model.eval()
    # This method merges the LoRa layers into the base model. 
    # This is needed if someone wants to use the base model as a standalone model.
    # if we quantize the model we actually can't use the merge
    model = model.merge_and_unload()
    
    # effiency
    # model = model.to_bettertransformer()

    
    
    num_cats = len(cat_dataset)
    
    cat_embeddings_array = np.zeros((num_cats, args.embed_dim))
    for step, batch in enumerate(tqdm(cat_dataloader)):
        with torch.no_grad():
            #with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                category_embs = model(**{k: v.to(args.device) for k, v in batch.items()}).detach().float().cpu()
        start_index = step * args.test_batch_size
        end_index = start_index + args.test_batch_size if (start_index + args.test_batch_size) < num_cats else num_cats
        cat_embeddings_array[start_index:end_index] = category_embs
        del category_embs, batch
    
    cat_search_index = construct_search_index(args.embed_dim, num_cats, cat_embeddings_array)
    
    
    # Initialize counters
    batch_precision = []
    batch_recall = []
    total_hit = 0
    
    for test_example in tokenized_test_titles:

        query_embeddings = get_query_embeddings(test_example["x_title"], model, tokenizer, args.max_length, args.device)
        search_results = get_nearest_neighbours(args.top_k, cat_search_index, query_embeddings, ids_to_cat_dict, threshold=0.0)
        
        
        retrieved_cats = [item[0] for item in search_results]
        correct_categories = list(set([value for key, value in test_example.items() if key.startswith('cat')]))
        
        
        precision, recall = calculate_precision_recall(retrieved_cats, correct_categories)
        
        batch_precision.append(precision)
        batch_recall.append(recall)
        
        hit = any(cat in retrieved_cats for cat in correct_categories)
        total_hit += hit

  
   
    total_examples = len(tokenized_test_titles)
    recall = sum(batch_recall) / total_examples
    precision = sum(batch_precision) / total_examples
    hit_rate = total_hit / float(total_examples)

    print("Recall:", recall)
    print("Precision:", precision)
    print("Hit Rate:", hit_rate)


    
    query = "machine learning engineer"
    k = args.top_k
    query_embeddings = get_query_embeddings(query, model, tokenizer, args.max_length, args.device)
    search_results = get_nearest_neighbours(k, cat_search_index, query_embeddings, ids_to_cat_dict, threshold=0.0)
    
    


    print(f"{query=}")
    for product, cosine_sim_score in search_results:
        print(f"cosine_sim_score={round(cosine_sim_score,2)} {product=}")
            
    
    
    
if __name__ == "__main__":
    main()
    
    

# python contrastive_train/test_with_hnsw.py --dataset_path "./dataset" --model_name_or_path "BAAI/bge-large-en" --peft_model_path "./sementic_search_outs"


# with 8 bit quantization: 

# Recall: 0.74673670805476
# Precision: 0.1546800382043926
# Hit Rate: 0.8500477554918816