from googletrans import Translator
import os
from parse_xml import get_sentences
import threading
import concurrent.futures
import time
import pandas as pd
import numpy as np

class Google_translator:
    def __init__(self, start, end) -> None:
        self.translator = Translator()
        self.start = start
        self.end = end
        self.translated_file = "./../Data/gigafida_translated.csv"
        self.original_file = "./../Data/gigafida_original.csv"
        self.path = "./../Data/ccGigafidaV1_0/"
        self.files = sorted(os.listdir(self.path))
        self.lock = threading.Lock()
        self.max_workers = 8
        self.separator = "~"
        self.chunk = 1000
        if not os.path.isfile("./../Data/gigafida_translated.csv"):
            f = open(self.translated_file, "w", encoding="utf-8")
            f.write("Fid" + self.separator +"Sid" +self.separator +"Text\n")
            f.close()
            f = open(self.original_file, "w", encoding="utf-8")
            f.write("Fid" + self.separator +"Sid" +self.separator +"Text\n")
            f.close()
            self.fid = start
            self.sid = 0
            self.from_before = []
        else:
            with open(self.translated_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            #self.fid = int(lines[-1].split(self.separator)[0])
            self.sid = int(lines[-1].split(self.separator)[1]) + 1
            l = pd.read_csv(self.original_file, sep=self.separator, on_bad_lines="skip")
            self.from_before = []
            self.fid = np.max(l.Fid.values) +1
            for i in range(start, np.max(l.Fid.values)):
                if i not in l.Fid.values:
                    self.from_before.append(i)
            print("Missing from before: ", self.from_before)
    
    def translate(self, sentence1):
        translation = self.translator.translate(sentence1, dest="en")
        pair = self.translator.translate(translation.text, dest="sl")
        return pair.text
    
    def write_in_file(self, file, text):
        f = open(file, "a", encoding="utf-8")
        f.write(str(self.fid) + self.separator + str(self.sid) + self.separator + text+"\n")
        f.close()
    
    def write_arrays(self, sentences, translated, fid):
        f = open(self.original_file, "a", encoding="utf-8")
        for i in range(len(sentences)):
            f.write(str(fid) + self.separator + str(i) + self.separator + sentences[i]+"\n")
        f.close()
        f = open(self.translated_file, "a", encoding="utf-8")
        for i in range(len(translated)):
            f.write(str(fid) + self.separator + str(i) + self.separator + translated[i]+"\n")
        f.close()

    def translate_all2(self):
        from_before = get_sentences(self.path + self.files[self.fid])
        print(self.sid, len(from_before))
        for i in range(self.sid, len(from_before)):
            self.sid = i
            translation = self.translate(from_before[i])
            self.write_in_file(self.original_file, from_before[i])
            self.write_in_file(self.translated_file, translation)
        self.fid += 1

        for f in range(self.fid, len(self.files)):
            print(f)
            self.fid = f
            sentences = get_sentences(self.path + self.files[f])
            for i in range(len(sentences)):
                self.sid = i
                translation = self.translate(sentences[i])
                self.write_in_file(self.original_file, sentences[i])
                self.write_in_file(self.translated_file, translation)
     
    def translate_all(self, f):
        done = False
        while not done:
            try:
                sentences = get_sentences(self.path + self.files[f])
                translated_sentences = []
                print("Thread: ", f, len(sentences))
                for i in range(len(sentences)):
                    done2 = False
                    while not done2:
                        try:
                            self.sid = i
                            translation = self.translate(sentences[i])
                            translated_sentences.append(translation)
                            done2 = True
                        except Exception as er:
                            print("error :", f, i, er)
                            time.sleep(1)
                with self.lock:
                    print("Writing: ", f, len(sentences))
                    self.write_arrays(sentences, translated_sentences, f)
                done = True
            except Exception as e:
                print(f, e)
                time.sleep(5)
                

    def translate_all_multithread(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for f in self.from_before:
                t = pool.submit(self.translate_all, f)
            for f in range(self.fid, self.end):
                t = pool.submit(self.translate_all, f)


def main():
    google_translator = Google_translator(0, 31722 //4)
    google_translator.translate_all_multithread()

main()