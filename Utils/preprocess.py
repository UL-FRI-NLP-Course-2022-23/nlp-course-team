import pandas as pd
import numpy as np

def read_csv(file):
    f = open(file, "r" ,encoding="utf-8")
    header = f.readline()
    fids = []
    sids = []
    text = []
    for line in f:
        col = line.split("~")
        fids.append(int(col[0]))
        sids.append(int(col[1]))
        sentence = "~".join(col[2:])
        if sentence[-1] == "\n":
            sentence = sentence[:-1]
        text.append(sentence)
    f.close()
    ori = pd.DataFrame()
    ori["Fid"] = fids
    ori["Sid"] = sids
    ori["Text"] = text
    return ori

def combine(original, translated):
    if original[["Fid", "Sid"]].equals(translated[["Fid", "Sid"]]):
        combined = pd.DataFrame()
        combined["input_text"] = original["Text"]
        combined["target_text"] = translated["Text"]
        return combined
    else:
        raise Exception("Error, sentences are not correctly aligned. More preprocessing is needed.")
    
def remove_same_translations(data):
    return data[data.input_text != data.target_text]