import os
from parse_xml import get_sentences
from nemo.core.classes.modelPT import ModelPT

class Nemo_translator:
    def __init__(self) -> None:
        self.slen = ModelPT.restore_from("./../Models/slen/aayn_base.nemo", map_location="cpu")
        self.ensl = ModelPT.restore_from("./../Models/ensl/aayn_base.nemo", map_location="cpu")
        self.translated_file = "./../Data/gigafida_translated.csv"
        self.original_file = "./../Data/gigafida_original.csv"
        self.path = "./../Data/ccGigafidaV1_0/"
        self.files = sorted(os.listdir(self.path))

        if not os.path.isfile("./../Data/gigafida_translated.csv"):
            f = open(self.translated_file, "w", encoding="utf-8")
            f.write("Fid;Sid;Text\n")
            f.close()
            f = open(self.original_file, "w", encoding="utf-8")
            f.write("Fid;Sid;Text\n")
            f.close()
            self.fid = 0
            self.sid = 0
        else:
            with open(self.translated_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.fid = int(lines[-1].split(";")[0])
            self.sid = int(lines[-1].split(";")[1]) + 1
    
    def translate(self, sentences):
        trans_en = self.slen.translate(sentences)
        trans_sl = self.ensl.translate(trans_en)
        return trans_sl
    
    def write_in_file(self, file, text):
        f = open(file, "a", encoding="utf-8")
        f.write(str(self.fid) + ";" + str(self.sid) + ";" + text+"\n")
        f.close()

    def translate_all(self):
        for f in range(self.fid, len(self.files)):
            print(f)
            self.fid = f
            sentences = get_sentences(self.path + self.files[f])
            pair = self.translate(sentences)
            for i in range(len(sentences)):
                self.write_in_file(self.original_file, sentences[i])
                self.write_in_file(self.translated_file, pair[i])
    
    def translate_all2(self):
        from_before = get_sentences(self.path + self.files[self.fid])
        print(self.sid, len(from_before))
        for i in range(self.sid, len(from_before)):
            self.sid = i
            translation = self.translate([from_before[i]])
            self.write_in_file(self.original_file, from_before[i])
            self.write_in_file(self.translated_file, translation[0])
        self.fid += 1

        for f in range(self.fid, len(self.files)):
            print(f)
            self.fid = f
            sentences = get_sentences(self.path + self.files[f])
            for i in range(len(sentences)):
                self.sid = i
                translation = self.translate([sentences[i]])
                self.write_in_file(self.original_file, sentences[i])
                self.write_in_file(self.translated_file, translation[0])
    
    def translate_all3(self):
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

def main():
    google_translator = Nemo_translator()
    google_translator.translate_all2()

main()