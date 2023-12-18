import nlpaug.augmenter.word as naw
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# sbatch -n 1 --cpus-per-task=4 --time=6:00:00 --job-name="learn1" --mem-per-cpu=16G --gpus=1 --gres=gpumem:16G --wrap "python3 text_complexity/script_data/data_aug.py"

if __name__ == '__main__':
    stop_words = stopwords.words('english')
    stop_words.extend(["I", "Hi", "I'll", "I'm", "don't", "doesn't"])
    train_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/train_pruned_efcamdat.csv')
    aug_syn = naw.SynonymAug(
        aug_src='wordnet', aug_max=2, stopwords=stop_words)
    aug_sub = naw.WordEmbsAug(
        model_type='fasttext', model_path='/cluster/work/sachan/abhinav/text_complexity/embedding/wiki-news-300d-1M.vec',
        action="substitute", stopwords=stop_words, aug_max=2)

    def syn_aug(sentence):
        aug_sentence = aug_syn.augment(sentence)
        return aug_sentence[0]

    def sub_aug(sentence):
        aug_sentence = aug_sub.augment(sentence)
        return aug_sentence[0]

    syn_aug_dataset_df = pd.DataFrame()
    sub_aug_dataset_df = pd.DataFrame()

    for cefr_numeric in range(1, 6):
        cefr_numeric_df = train_df[train_df['cefr_numeric'] == cefr_numeric]
        # get 25% of the cefr_numeric_df randomly and add it to testt_df
        _, test = train_test_split(
            cefr_numeric_df, test_size=0.25, random_state=42)
        test['syn_aug'] = test['sentences'].apply(lambda x: syn_aug(x))
        test['sub_aug'] = test['sentences'].apply(lambda x: sub_aug(x))
        syn_aug_dataset_df = pd.concat([syn_aug_dataset_df, test])
        sub_aug_dataset_df = pd.concat([sub_aug_dataset_df, test])

    syn_aug_dataset_df.to_csv(
        "/cluster/work/sachan/abhinav/text_complexity/data/syn_aug_125.csv", index=False)
    sub_aug_dataset_df.to_csv(
        "/cluster/work/sachan/abhinav/text_complexity/data/sub_aug_fastext_125.csv", index=False)
    syn_aug_dataset_df.drop('sentences', axis=1, inplace=True)
    syn_aug_dataset_df = syn_aug_dataset_df.rename(
        columns={'syn_aug': 'sentences'})
    train_syn_aug_dataset_df = pd.concat([train_df, syn_aug_dataset_df])
    train_syn_aug_dataset_df.to_csv(
        "/cluster/work/sachan/abhinav/text_complexity/data/train_syn_aug_125.csv", index=False)
    sub_aug_dataset_df.drop('sentences', axis=1, inplace=True)
    sub_aug_dataset_df = sub_aug_dataset_df.rename(
        columns={'sub_aug': 'sentences'})
    train_sub_aug_dataset_df = pd.concat([train_df, sub_aug_dataset_df])
    train_sub_aug_dataset_df.to_csv(
        "/cluster/work/sachan/abhinav/text_complexity/data/train_sub_aug_fastext_125.csv", index=False)
