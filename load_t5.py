from simpletransformers.t5 import T5Model
import os

root_dir = os.getcwd()
trained_model_path = os.path.join(root_dir,"outputs")
testing_sentences = ["To je test.", 
                     "Kot razlog je navedel, da je bil lačen, ker je izpustil zajtrk.",
                     "Po Sloveniji so zagoreli številni kresovi.",
                     "Ameriške finančne oblasti so zasegle kalifornijsko banko First Republic.",
                     "Leonard je Penny zaželel srečo!",
                     "SpaceX je v vesolje uspešno izstrelil svojo komercialno raketo.",
                     "Sheldon je rekel: 'Za kosilo želim jesti kitajsko hrano.'"
                     ]

args = {
"overwrite_output_dir": True,
"max_seq_length": 256,
"max_length": 50,
"top_k": 50,
"top_p": 0.95,
"num_return_sequences": 10,
"num_beams": 1
}
trained_model = T5Model("t5",trained_model_path,args=args)

prefix = "paraphrase"
"""print(f"{prefix}: To je test.")
pred = trained_model.predict([f"{prefix}: Danes je lep sončen dan."])
print(pred)"""
for sentence in testing_sentences:
    print("---------")
    pred= trained_model.predict([f"{prefix}: {sentence}"])
    print(sentence)
    for a in pred[0]:
        print(a)