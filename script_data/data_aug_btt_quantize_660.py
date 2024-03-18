import pdb

import ctranslate2
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from nltk.translate.bleu_score import sentence_bleu
from sacrebleu.metrics import BLEU
from tqdm import tqdm

bleu = BLEU()

train_25_aug = pd.read_csv(
    "/cluster/work/sachan/abhinav/text_complexity/exp_data/split/train_3.csv")

train_25_aug = train_25_aug[['writing_id', 'sentences', 'cefr_numeric']]

# bcp 47 code https://github.com/facebookresearch/flores/blob/main/toxicity/README.md
list_of_translation_lang = ['zho_Hans', 'spa_Latn', 'arb_Arab', 'fra_Latn',
                            'deu_Latn', 'rus_Cyrl', 'jpn_Jpan', 'por_Latn', 'hin_Deva', 'ita_Latn']

ct_model_path = "/cluster/work/sachan/abhinav/model/ctranslate/nllb-200-distilled-600M-int8"
spm_model_path = "/cluster/work/sachan/abhinav/model/ctranslate/flores200_sacrebleu_tokenizer_spm.model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

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
torch.cuda.empty_cache()
sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)

translator = ctranslate2.Translator(ct_model_path, device)
beam_size = 4


data_rows = []
data_rows_all = []
chunk_size = 200
chunked_df = [group for _, group in train_25_aug.groupby(
    np.arange(len(train_25_aug)) // chunk_size)]

print("Total chunks: ", len(chunked_df))
all_df = pd.DataFrame()
for chunk in tqdm(chunked_df, total=len(chunked_df)):
    sentences = chunk['sentences'].astype(str).tolist()
    collected_data = [[]]*len(list_of_translation_lang)
    for idx, tgt_lang in enumerate(list_of_translation_lang):
        src_lang = "eng_Latn"
        torch.cuda.empty_cache()
        target_prefix = [[tgt_lang]] * chunk_size
        source_sents_encoded = sp.encode_as_pieces(sentences)
        source_sents_subworded = [[src_lang] + sent + ["</s>"]
                                  for sent in source_sents_encoded]

        # Translate
        translations_result = translator.translate_batch(source_sents_subworded,
                                                         target_prefix=target_prefix,
                                                         beam_size=beam_size,
                                                         max_batch_size=2048,
                                                         disable_unk=True)

        # torch.cuda.empty_cache()
        translation_subworded = [translation.hypotheses[0]
                                 for translation in translations_result]
        for translation in translation_subworded:
            if tgt_lang in translation:
                translation.remove(tgt_lang)

        translations = sp.decode(translation_subworded)

        # Back-translation
        target_prefix = [[src_lang]] * chunk_size
        translation_sent_encoded = sp.encode_as_pieces(translations)
        translation_sent_subworded = [[tgt_lang] +
                                      sent + ["</s>"] for sent in translation_sent_encoded]
        back_translations_result = translator.translate_batch(translation_sent_subworded,
                                                              target_prefix=target_prefix,
                                                              beam_size=beam_size,
                                                              max_batch_size=2048,
                                                              disable_unk=True)
        # torch.cuda.empty_cache()
        back_translations_subworded = [translation.hypotheses[0]
                                       for translation in back_translations_result]
        for back_translation in back_translations_subworded:
            if src_lang in back_translation:
                back_translation.remove(src_lang)
        back_translations = sp.decode(back_translations_subworded)
        collected_data[idx] = back_translations

    # merge collected_data to chunk
    # pdb.set_trace()
    chunk['zho_sent'] = collected_data[0]
    chunk['spa_sent'] = collected_data[1]
    chunk['arb_sent'] = collected_data[2]
    chunk['fra_sent'] = collected_data[3]
    chunk['deu_sent'] = collected_data[4]
    chunk['rus_sent'] = collected_data[5]
    chunk['jpn_sent'] = collected_data[6]
    chunk['por_sent'] = collected_data[7]
    chunk['hin_sent'] = collected_data[8]
    chunk['ita_sent'] = collected_data[9]
    all_df = pd.concat([all_df, chunk])

all_df.to_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/train_3.csv', index=False)
# srun -n 1 --cpus-per-task=4 --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --gpus=1 --gres=gpumem:24G --pty python3 text_complexity/script_data/data_aug_btt.py
