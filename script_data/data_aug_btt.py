import itertools
import pdb

import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import BLEU
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer,
                          pipeline)

bleu = BLEU()

# Reference and hypothesis
# ref = [["The cat is on the mat"]]
# candidate = ["The cat is on the mat"]
# https://github.com/mjpost/sacrebleu?tab=readme-ov-file#using-sacrebleu-from-python

# Calculate BLEU score
# bleu_score = bleu.corpus_score(candidate, [ref])
# print(bleu_score.score)

# Calculate BLEU score using nltk
# ref1 = ["The cat is on the mat".split()]
# candidate1 = "The cat is on the mat".split()
# weight for n-grams
# 1-gram = (1,0,0,0)
# 2-gram = (0,1,0,0)
# 3-gram = (1,0,1,0)
# 4-gram = (0,0,0,1)
# https://www.digitalocean.com/community/tutorials/bleu-score-in-python
# print(sentence_bleu(ref1, candidate1))

# Use a pipeline as a high-level helper
# pipe_en_to_fr = pipeline("translation_en_to_fr",
#                          model="facebook/nllb-200-1.3B", device=device, framework='pt')
# pipe_fr_to_en = pipeline("translation_fr_to_en", model="facebook/nllb-200-1.3B", device=device,
#                          framework='pt',)

train_25_aug = pd.read_csv(
    "/cluster/work/sachan/abhinav/text_complexity/exp_data/additional_0.75.csv")

train_25_aug = train_25_aug[['writing_id', 'sentences', 'cefr_numeric']]

# bcp 47 code https://github.com/facebookresearch/flores/blob/main/toxicity/README.md
list_of_translation_lang = ['zho_Hans', 'spa_Latn', 'arb_Arab', 'fra_Latn', 'deu_Latn',
                            'rus_Cyrl', 'jpn_Jpan', 'por_Latn', 'hin_Deva', 'ita_Latn']

# Chinese (Mandarin)
# Spanish
# Arabic
# French
# German
# Russian
# Japanese
# Portuguese
# Hindi
# Italian


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_eng = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="eng_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_zho = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="zho_Hans",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_spa = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="spa_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_arb = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="arb_Arab",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_fra = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="fra_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_deu = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="deu_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_rus = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="rus_Cyrl",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_jpn = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="jpn_Jpan",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_por = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="por_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_hin = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="hin_Deva",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
tokenizer_ita = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="ita_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)

tokenizer_dict = {
    "zho_Hans": tokenizer_zho,
    "spa_Latn": tokenizer_spa,
    "arb_Arab": tokenizer_arb,
    "fra_Latn": tokenizer_fra,
    "deu_Latn": tokenizer_deu,
    "rus_Cyrl": tokenizer_rus,
    "jpn_Jpan": tokenizer_jpn,
    "por_Latn": tokenizer_por,
    "hin_Deva": tokenizer_hin,
    "ita_Latn": tokenizer_ita,
    "eng_Latn": tokenizer_eng
}

# https://huggingface.co/facebook/nllb-200-1.3B/tree/main
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-1.3B", cache_dir="/cluster/work/sachan/abhinav/model/fb/")
model.to(device)

data_rows = []
data_rows_all = []
for _, row in train_25_aug.iterrows():
    sentence = row['sentences']
    ref = [[sentence]]
    input_tokenize_eng = tokenizer_eng(sentence, return_tensors="pt")
    input_tokenize_eng.to(device)
    highest_bleu = float('-inf')
    second_highest_bleu = float('-inf')
    highest_lang_sent = second_highest_lang_sent = ''
    highest_lang = second_highest_lang = ''
    data_all_lang = []
    for translation_lang in list_of_translation_lang:
        sentence_tokens = model.generate(
            **input_tokenize_eng, forced_bos_token_id=tokenizer_eng.lang_code_to_id[translation_lang], max_length=200)
        translated_tokenizer = tokenizer_dict[translation_lang]
        translated_sentence = translated_tokenizer.batch_decode(
            sentence_tokens, skip_special_tokens=True)[0]
        translated_input = translated_tokenizer(
            translated_sentence, return_tensors="pt")
        translated_input.to(device)
        eng_translated_tokens = model.generate(
            **translated_input, forced_bos_token_id=tokenizer_eng.lang_code_to_id['eng_Latn'], max_length=200)
        eng_translated_sentence = tokenizer_eng.batch_decode(
            eng_translated_tokens, skip_special_tokens=True)[0]
        candidate = [eng_translated_sentence]
        data_all_lang.extend([eng_translated_sentence])
        bleu_score = bleu.corpus_score(candidate, ref)
        bleu_score = bleu_score.score
        if bleu_score > highest_bleu:
            second_highest_bleu = highest_bleu
            second_highest_lang = highest_lang
            second_highest_lang_sent = highest_lang_sent
            highest_bleu = bleu_score
            highest_lang = translation_lang
            highest_lang_sent = eng_translated_sentence
        elif bleu_score > second_highest_bleu:
            second_highest_bleu = bleu_score
            second_highest_lang = translation_lang
            second_highest_lang_sent = eng_translated_sentence
        # print('translation_lang', translation_lang)
        # print('translated_sentence', translated_sentence)
        # print('tokens', input_tokenize_eng,
        #       translated_input, eng_translated_tokens)
        # print('eng_translated_sentence', eng_translated_sentence)
        # print('bleu_score', bleu_score)

    data = [row['writing_id'], sentence, row['cefr_numeric'], highest_lang_sent,
            highest_lang, second_highest_lang_sent, second_highest_lang]
    # data_all_lang = list(itertools.chain.from_iterable(data_all_lang))
    data_all = [row['writing_id'], sentence, row['cefr_numeric']]
    data_all.extend(data_all_lang)
    data_rows.append(data)
    data_rows_all.append(data_all)
data_df = pd.DataFrame(data_rows, columns=['writing_id', 'sentences', 'cefr_numeric',
                       'highest_lang_sent', 'highest_lang', 'second_highest_lang_sent', 'second_highest_lang'])

data_all_df = pd.DataFrame(data_rows_all, columns=['writing_id', 'sentences', 'cefr_numeric',
                                                   'zho_sent', 'spa_sent', 'arb_sent', 'fra_sent', 'deu_sent', 'rus_sent', 'jpn_sent', 'por_sent', 'hin_sent', 'ita_sent'])

data_df.to_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/translation/additional0.75_bleu.csv', index=False)

data_all_df.to_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/translation/additional0.75_all.csv', index=False)

# srun -n 1 --cpus-per-task=4 --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --gpus=1 --gres=gpumem:24G --pty python3 text_complexity/script_data/data_aug_btt.py
