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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_eng = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="eng_Latn",
                                              cache_dir="/cluster/work/sachan/abhinav/model/fb/",
                                              device=device)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-1.3B", cache_dir="/cluster/work/sachan/abhinav/model/fb/")
model.to(device)

article = "Information about training algorithms, parameters, fairness constraints or other applied approaches, and features."
ref = [[article]]
inputs = tokenizer_eng(article, return_tensors="pt")
inputs.to(device)

list_of_translation_lang = ["deu_Latn", "fra_Latn", "ita_Latn", "por_Latn", "spa_Latn", "rus_Cyrl",
                            "ara_Arab", "zho_Hans", "jpn_Hani", "tur_Latn", "heb_Hebr", "hin_Deva", "ben_Beng"]

for translation_lang in list_of_translation_lang:
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer_eng.lang_code_to_id[translation_lang], max_length=30)
    print(translation_lang)
    print(tokenizer_eng.batch_decode(
        translated_tokens, skip_special_tokens=True)[0])
    translated_article = tokenizer_eng.batch_decode(
        translated_tokens, skip_special_tokens=True)[0]
    candidate = [translated_article]
    bleu_score = bleu.corpus_score(candidate, ref)
    print(bleu_score.score)
    print("\n")

# srun -n 1 --cpus-per-task=4 --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --gpus=1 --gres=gpumem:24G --pty python3 text_complexity/script_data/data_aug_btt.py
