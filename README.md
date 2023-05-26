# Natural language processing course 2022/23: `Project 3: Paraphrasing sentences`

Team members:
 * `Matej Vatovec`, `63190310`, `mv6299@student.uni-lj.si`
 * `Rok Cek`, `63190074`, `rc2136@student.uni-lj.si`
 * `MEMBER FULL NAME`, `STUDENT ID`, `STUDENT E-MAIL`
 
Group public acronym/name: `THINK OF PUBLIC STRING FOR YOUR GROUP`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Author
Matej Vatovec

## How to Set Up

Follow the instructions below to set up the environment:

```sh
conda env create --name st --file=environment.yml
conda activate st
```

## How to use:
Back-translation:
```sh
python Utils/translate_google.py
```
Model fine tuning:

```sh
python train4_t5.py
```

Visualize paraphrases for manual evaluation:

```sh
python manual_eval_t5.py
```

Automatic evaluation:
```sh
python automatic_eval_t5.py
```
## Additional resources
Our dataset is available here: https://drive.google.com/file/d/1aLUCF1dYSJS7UdrYPA6I_5JcZgPeQHSY/view?usp=sharing

Our fine tuned model is available here: https://drive.google.com/file/d/1ACogbdv-07fELMXjNb7zUqnkF4aUBYGy/view?usp=sharing

The model folder should be located in ./Models