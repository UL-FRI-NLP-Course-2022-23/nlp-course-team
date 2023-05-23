from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Utils.sentence_paraphrases import *
from nltk.translate.bleu_score import sentence_bleu
import string
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

data = pd.read_csv("./final.csv")
data = data.iloc[-1000:].reset_index()

tokenizer = AutoTokenizer.from_pretrained("Models/t5_model", truncate=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Models/t5_model").to("cuda")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

prefix = "paraphrase: "
bleu_scores = 0
rougel = 0
rouge1 = 0

for i in tqdm(range(len(data))):
    print("---------")
    a = tokenizer(prefix + data.iloc[i]["input_text"], padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids = a.input_ids.to("cuda")
    outputs = model.generate(input_ids,
                            max_length=256,
                            no_repeat_ngram_size=2,
                            num_beams=1,
                            num_return_sequences=1,
                            )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    bleu_scores += sentence_bleu(data.iloc[i]["target_text"], decoded[0])
    rougel += scorer.score(data.iloc[i]["target_text"], decoded[0])["rougeL"][1]
    rouge1 += scorer.score(data.iloc[i]["target_text"], decoded[0])["rouge1"][1]
print("Bleu: ", bleu_scores / 1000)
print("Rouge-L: ", rougel/1000)
print("Rouge1: ", rouge1/1000)
