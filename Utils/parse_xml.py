from bs4 import BeautifulSoup

def get_sentences(file):
    with open(file, 'r', encoding="utf-8") as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "xml")
    body = Bs_data.find('body')
    sentences = []
    for p in body.find_all("p"):
        for s in p.find_all("s"):
            sentence = ""
            for i in s.children:
                if i.name == "w" or i.name == "c":
                    sentence += i.text
                elif i.name == "S":
                    sentence += " "
            sentences.append(sentence)
    return sentences