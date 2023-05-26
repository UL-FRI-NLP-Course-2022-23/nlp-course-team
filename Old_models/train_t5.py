import pandas as pd
from sklearn.model_selection import train_test_split
import Utils.preprocess as preprocess
from simpletransformers.t5 import T5Model
from transformers import T5Tokenizer
import sklearn
import os
from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained("cjvt/t5-sl-small")
args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "num_train_epochs": 1,
    "num_beams": None,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "use_multiprocessing": False,
    "save_steps": -1,
    "save_eval_checkpoints": True,
    "evaluate_during_training": False,
    "adam_epsilon": 1e-08,
    "eval_batch_size": 4,
    "fp_16": False,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0003,
    "max_grad_norm": 1.0,
    "n_gpu": 1,
    "seed": 42,
    "train_batch_size": 50,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "tokenizer": tokenizer,
}

#data = preprocess.combine(preprocess.read_csv("./Data/gigafida_original.csv"), preprocess.read_csv("./Data/gigafida_translated.csv"))
#data = preprocess.remove_same_translations(data)
#data["prefix"] = "paraphrase"#
all_data = []
for file in os.listdir("./Data/vm/"):
    data = preprocess.combine(preprocess.read_csv("./Data/vm/" + file +"/gigafida_original.csv"), preprocess.read_csv("./Data/vm/" + file +"/gigafida_translated.csv"))
    data = preprocess.remove_same_translations(data)
    data["prefix"] = "paraphrase"
    all_data.append(data)
data = pd.concat(all_data, ignore_index=True)
data = data.dropna()
train_data,test_data = train_test_split(data,test_size=0.1)

print("load")
#data = load_dataset("csv", data_files="final.csv", split="train")
#data = data.train_test_split(0.1)
print("Definition")
model = T5Model("t5","cjvt/t5-sl-small", args=args, tokenizer=tokenizer)
print("Train")
#model.train_model(data["train"], eval_data=data["test"], use_cuda=True,acc=sklearn.metrics.accuracy_score)
model.train_model(train_data, eval_data=test_data, use_cuda=True,acc=sklearn.metrics.accuracy_score)
