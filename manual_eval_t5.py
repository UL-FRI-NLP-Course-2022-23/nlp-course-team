from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Utils.sentence_paraphrases import *
from nltk.translate.bleu_score import sentence_bleu
import string
#from rouge_score import rouge_score

tokenizer = AutoTokenizer.from_pretrained("Models/t5_model", truncate=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Models/t5_model").to("cuda")

prefix = "paraphrase: "
sentence = "SpaceX je v vesolje uspešno izstrelil svojo komercialno raketo."

testing_sentences = ["To je test.", 
                     "Kot razlog je navedel, da je bil lačen, ker je izpustil zajtrk.",
                     "Po Sloveniji so zagoreli številni kresovi.",
                     #"Ameriške finančne oblasti so zasegle kalifornijsko banko First Republic.",
                     "Leonard je Penny zaželel srečo!",
                     "SpaceX je v vesolje uspešno izstrelil svojo komercialno raketo.",
                     "Sheldon je rekel: 'Za kosilo želim jesti kitajsko hrano.'"
                     ]

references = [GROUP1, GROUP2, GROUP3, GROUP4, GROUP5]

for i in range(len(testing_sentences)):
    print("---------")
    print(testing_sentences[i])
    a = tokenizer(prefix + testing_sentences[i], padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids = a.input_ids.to("cuda")
    outputs = model.generate(input_ids,
                            max_length=256,
                            no_repeat_ngram_size=2,
                            num_beams=10,
                            num_return_sequences=10,
                            num_beam_groups=5,
                            diversity_penalty=0.1
                            )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for item in decoded:
        print(item)
        #print(sentence_bleu(references[i], item.translate(str.maketrans('', '', string.punctuation)).lower().split(" ")))
    #for l in outputs:
        #print(tokenizer.batch_decode(outputs[l], skip_special_tokens=True))
