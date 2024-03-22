import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU
from sklearn.model_selection import train_test_split
from tqdm import tqdm

bleu = BLEU()

df_0 = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/train_0.csv')
df_1 = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/train_1.csv')
df_2 = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/train_2.csv')
df_3 = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/train_3.csv')
df_4 = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/train_4.csv')
print(df_0.shape, df_1.shape, df_2.shape, df_3.shape, df_4.shape)

df = pd.concat([df_0, df_1, df_2, df_3, df_4])
print(df.shape)

col_name = ['zho_sent', 'spa_sent', 'arb_sent', 'fra_sent', 'deu_sent',
            'rus_sent', 'jpn_sent', 'por_sent', 'hin_sent', 'ita_sent']


def calculate_highest_bleu_score(row):
    sentence = row['sentences']
    ref = [[sentence]]
    highest_bleu_score = 0
    highest_lang = ''
    highest_bleu_sent = ''
    for col in col_name:
        lang = col.split('_')[0]
        sent = row[col]
        candidate = [sent]
        bleu_score = bleu.corpus_score(candidate, ref)
        if bleu_score.score > highest_bleu_score:
            highest_bleu_score = bleu_score.score
            highest_lang = lang
            highest_bleu_sent = sent
    return highest_lang, highest_bleu_sent


# write function to select rand col from col_name and then return sentence from that col
def select_rand_col(row):
    col = np.random.choice(col_name)
    return row[col]


def generate_given_dataset(df, percentage):
    train_df = pd.DataFrame()
    # select given percentage of data from each cefr_numeric value
    for cefr_numeric in range(1, 6):
        cefr_numeric_df = df[df['cefr_numeric'] == cefr_numeric]
        # select given percentage of cefr_numeric_df randomly and add it to train_df
        train, _ = train_test_split(
            cefr_numeric_df, train_size=percentage, random_state=42)
        train_df = pd.concat([train_df, train])

    print(percentage, df.shape, percentage * df.shape[0])
    print(train_df.shape)
    print('\n')
    train_df.to_csv(
        f"/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/bleu/additional_{percentage}.csv", index=False)


def generate_given_dataset1(df, percentage):
    train_df = pd.DataFrame()
    # select given percentage of data from each cefr_numeric value
    for cefr_numeric in range(1, 6):
        cefr_numeric_df = df[df['cefr_numeric'] == cefr_numeric]
        # select given percentage of cefr_numeric_df randomly and add it to train_df
        train, _ = train_test_split(
            cefr_numeric_df, train_size=percentage, random_state=42)
        train_df = pd.concat([train_df, train])

    print(percentage, df.shape, percentage * df.shape[0])
    print(train_df.shape)
    print('\n')
    train_df.to_csv(
        f"/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/random/additional_{percentage}.csv", index=False)


data_all = []
print('processing data')
# for _, row in df.iterrows():
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    lang, sent = calculate_highest_bleu_score(row)
    data_all.append([row['writing_id'], row['sentences'],
                    row['cefr_numeric'], sent, lang])

df_all = pd.DataFrame(data_all, columns=[
                      'writing_id', 'sentences', 'cefr_numeric', 'translated_sentence', 'translated_lang'])

df_all.to_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/bleu/train_all.csv', index=False)

generate_given_dataset(df_all, 0.05)
generate_given_dataset(df_all, 0.25)
generate_given_dataset(df_all, 0.50)
generate_given_dataset(df_all, 0.75)

data_all = []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    sent = select_rand_col(row)
    data_all.append([row['writing_id'], row['sentences'],
                    row['cefr_numeric'], sent])

df_all = pd.DataFrame(data_all, columns=[
    'writing_id', 'sentences', 'cefr_numeric', 'translated_sentence'])

df_all.to_csv(
    '/cluster/work/sachan/abhinav/text_complexity/exp_data/tt_nllb/random/train_all.csv', index=False)

generate_given_dataset1(df_all, 0.05)
generate_given_dataset1(df_all, 0.25)
generate_given_dataset1(df_all, 0.50)
generate_given_dataset1(df_all, 0.75)
