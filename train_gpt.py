from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, Trainer, AutoModelForCausalLM, Seq2SeqTrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import nltk

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
      predictions = predictions[0]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def compute_metrics2(eval_preds):
  preds, labels = eval_preds
  if isinstance(preds, tuple):
      preds = preds[0]
  decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
  if data_args.ignore_pad_token_for_loss:
      # Replace -100 in the labels as we can't decode them.
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  # Some simple post-processing
  decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

  result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  # Extract a few results from ROUGE
  result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
  result["gen_len"] = np.mean(prediction_lens)
  result = {k: round(v, 4) for k, v in result.items()}
  return result

metric = load_metric("rouge")
dataset = load_dataset("csv", data_files="final.csv", split="train")
dataset = dataset.remove_columns("Unnamed: 0")

tokenizer = AutoTokenizer.from_pretrained("cjvt/gpt-sl-base", truncate=True)
small = dataset.select((range(500000)))
small = small.remove_columns("prefix")

max_input_length = 256
max_target_length = 256

prefix="paraphrase: "
def preprocess_function(examples):
    #inputs = [prefix + doc for doc in examples["input_text"]]
    model_inputs = tokenizer(prefix + examples["input_text"], padding="max_length", max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], padding="max_length", max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_small = small.map(preprocess_function)
tokenized_small = tokenized_small.train_test_split(0.0001)


model = AutoModelForCausalLM.from_pretrained("cjvt/gpt-sl-base")
batch_size = 5
args = Seq2SeqTrainingArguments(
    output_dir="gpt_trainer",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False,
)

trainer = Trainer(
    model=model,
    #args=training_args,
    args=args,
    train_dataset=tokenized_small["train"],
    eval_dataset=tokenized_small["test"],
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics,
    )

trainer.train()
trainer.save_model("gpt_model")