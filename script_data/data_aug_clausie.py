import string

import claucy
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
claucy.add_to_pipe(nlp)
alpha = 0.45


def clausie_by_spacy(df):
    count_0, count_1 = 0, 0
    data_rows = []
    for _, row in df.iterrows():
        sentence = row['sentences']
        doc = nlp(sentence)
        clauses = doc._.clauses
        if len(clauses) == 0:
            count_0 += 1
            continue
        try:
            propositions = doc._.clauses[0].to_propositions(as_text=True)
        except:
            # print('error in sentence:', sentence)
            continue
        if len(propositions) == 1:
            count_1 += 1
        for prop in propositions:
            data = []
            character_ratio = len(prop) / len(sentence)
            if character_ratio > alpha:
                data = [row['writing_id'], prop, row['cefr_numeric']]
                data_rows.append(data)
            else:
                cefr_numeric = row['cefr_numeric']
                cefr_numeric = int(cefr_numeric)
                if cefr_numeric != 1:
                    cefr_numeric = cefr_numeric - 1
                data = [row['writing_id'], prop, cefr_numeric]
                data_rows.append(data)

    data_df = pd.DataFrame(data_rows, columns=[
                           'writing_id', 'sentences', 'cefr_numeric'])
    data_df.to_csv(
        '/cluster/work/sachan/abhinav/text_complexity/exp_data/clausie/additional_0.05.csv', index=False)
    # print('number of sentence without modification', count_0)
    # print('number of sentence with one extraction', count_1)


if __name__ == "__main__":
    train_25_aug = pd.read_csv(
        "/cluster/work/sachan/abhinav/text_complexity/exp_data/additional_0.05.csv")

    train_25_aug = train_25_aug[['writing_id', 'sentences', 'cefr_numeric']]
    clausie_by_spacy(train_25_aug)
