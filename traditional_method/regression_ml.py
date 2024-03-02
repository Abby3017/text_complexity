import logging
import pdb
from math import log

import nltk
import numpy as np
import pandas as pd
from numpy import mean, std
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import (LeaveOneOut, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, filename='/cluster/work/sachan/abhinav/text_complexity/traditional_method/regression_ml_with_diff_sub.log',
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def syllable_count(sentence):
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)
    total_syllables = 0
    vowels = "aeiouy"
    for word in words:
        count = 0
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        total_syllables += count
    return total_syllables


def uniq_tokens(sentence):
    words = nltk.word_tokenize(sentence)
    unique_tokens = set(words)
    return len(unique_tokens)

# token_type_ratio


def calculate_ttr(sentence):
    words = nltk.word_tokenize(sentence)
    num_unique_words = len(set(words))
    total_words = len(words)
    ttr = num_unique_words / total_words
    return ttr


# Lexical density is a measure of the proportion of content words (nouns, adjectives, verbs) in a sentence.


def calculate_density(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    open_wordclasses = ['NN', 'NNS', 'JJ', 'JJR', 'JJS',
                        'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ']
    num_content_words = sum(
        1 for word in pos_tags if word[1] in open_wordclasses)
    density = num_content_words / len(words)
    return density

# The Uber index ranges from 0 to 1.
# An index of 0 means all the words are the same, while an index of 1 means every word in the text is unique


def calculate_uber_index(text):
    words = nltk.word_tokenize(text)
    vocabulary = len(set(words))
    number_of_tokens = len(words)
    uber_index = log(number_of_tokens)**2 / \
        ((log(number_of_tokens/vocabulary)) + 1e-4)
    return uber_index

# count number of noun, adjective, verb, adverb in a sentence


def calculate_pos(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    noun_count, adjective_count, verb_count, adverb_count = 0, 0, 0, 0
    for word, tag in pos_tags:
        if tag.startswith('N'):
            noun_count += 1
        elif tag.startswith('J'):
            adjective_count += 1
        elif tag.startswith('V'):
            verb_count += 1
        elif tag.startswith('R'):
            adverb_count += 1
    # pos_counts = Counter(tag for word,tag in pos_tags)
    return {'noun_count': noun_count, 'adjective_count': adjective_count, 'verb_count': verb_count, 'adverb_count': adverb_count}


df_scores = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/data/data_aug_with_label_scores.csv')
df_complexity_scores = pd.read_csv(
    '/cluster/work/sachan/abhinav/text_complexity/data/data_aug_with_label_complexity_scores.csv')

merge_df = pd.merge(df_scores, df_complexity_scores, on='Unnamed: 0')

merge_df = merge_df[['sentences_x', 'sub_aug_x', 'word_freq_sentence', 'word_freq_sub',
                     'complexity_score', 'complexity_score_sub', 'bert_score_sub', 'cefr_numeric_x', 'cefr_numeric_sub_x']]

# get number of words using nltk tokenizer
merge_df['num_words'] = merge_df['sub_aug_x'].apply(
    lambda x: len(nltk.word_tokenize(x)))
# get average word length using nltk tokenizer
merge_df['avg_word_length'] = merge_df['sub_aug_x'].apply(
    lambda x: np.mean([len(word) for word in nltk.word_tokenize(x)]))
# count number of syllables in sentences
merge_df['num_syllables'] = merge_df['sub_aug_x'].apply(
    lambda x: syllable_count(x))
# get number of unique tokens in sentences
merge_df['num_unique_tokens'] = merge_df['sub_aug_x'].apply(
    lambda x: uniq_tokens(x))
# get type token ratio
merge_df['ttr'] = merge_df['sub_aug_x'].apply(lambda x: calculate_ttr(x))
# get lexical density
merge_df['lexical_density'] = merge_df['sub_aug_x'].apply(
    lambda x: calculate_density(x))
# get uber index
merge_df['uber_index'] = merge_df['sub_aug_x'].apply(
    lambda x: calculate_uber_index(x))
# average word length in syllables
merge_df['avg_word_length_syllables'] = merge_df['num_syllables'] / \
    merge_df['num_words']
# calculate_pos return dictionary with noun, adjective, verb, adverb counts in a sentence move them to separate columns for each sentence
merge_df[['noun_count', 'adjective_count', 'verb_count', 'adverb_count']] = pd.DataFrame(
    merge_df['sub_aug_x'].apply(lambda x: calculate_pos(x)).tolist(), index=merge_df.index)

merge_df.reset_index(drop=True, inplace=True)

y_train_change = merge_df['cefr_numeric_sub_x']
X_train = merge_df.drop(
    ['sub_aug_x', 'sentences_x', 'cefr_numeric_sub_x'], axis=1)

logging.info(X_train.columns.tolist())

scaler = StandardScaler()
scaler.fit(X_train)
X = scaler.transform(X_train)

lr_model = linear_model.LinearRegression()
ridge_model = linear_model.Ridge(alpha=0.5, random_state=1)
bayesian_model = linear_model.BayesianRidge()
gbr_model = GradientBoostingRegressor(random_state=1, loss='huber')

root_mean_scorer = make_scorer(
    root_mean_squared_error, greater_is_better=False)

cv = LeaveOneOut()
lr_predictions = []
ridge_predictions = []
bayesian_predictions = []
gbr_predictions = []
cefr_values = []

for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_train_change[train_index], y_train_change[test_index]
    cefr_values.append(y_test.values[0])
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_pred = np.round(y_pred)
    pred = y_pred[0]
    lr_predictions.append(pred)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    y_pred = np.round(y_pred)
    pred = y_pred[0]
    ridge_predictions.append(pred)
    bayesian_model.fit(X_train, y_train)
    y_pred = bayesian_model.predict(X_test)
    y_pred = np.round(y_pred)
    pred = y_pred[0]
    bayesian_predictions.append(pred)
    gbr_model.fit(X_train, y_train)
    y_pred = gbr_model.predict(X_test)
    y_pred = np.round(y_pred)
    pred = y_pred[0]
    gbr_predictions.append(pred)

lr_predictions = np.array(lr_predictions)
ridge_predictions = np.array(ridge_predictions)
bayesian_predictions = np.array(bayesian_predictions)
gbr_predictions = np.array(gbr_predictions)
cefr_values = np.array(cefr_values)

rmse = root_mean_squared_error(cefr_values, lr_predictions)
logging.info('Linear Regression rmse Score: %.3f' % rmse)
rmse = root_mean_squared_error(cefr_values, ridge_predictions)
logging.info('Ridge model rmse Score: %.3f' % rmse)
rmse = root_mean_squared_error(cefr_values, bayesian_predictions)
logging.info('Bayesian model rmse Score: %.3f' % rmse)
rmse = root_mean_squared_error(cefr_values, gbr_predictions)
logging.info('Gradient Boosting rmse Score: %.3f' % rmse)

exit()


scores = cross_val_score(lr_model, X, y_train,
                         scoring=root_mean_scorer, cv=cv, n_jobs=-1)
logging.info('Linear Regression loocv root_mean_squared_error: %.3f (%.3f)' %
             (mean(scores), std(scores)))

# lr_model.fit(X_n, train_y)
# y_pred = lr_model.predict(test_n)
# rmse = root_mean_squared_error(test_y, y_pred)
# logging.info('Linear Regression rmse Score: %.3f' % rmse)

cv = LeaveOneOut()
scores = cross_val_score(ridge_model, X, y_train,
                         scoring=root_mean_scorer, cv=cv, n_jobs=-1)
logging.info('Ridge model loocv root_mean_squared_error: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# ridge_model.fit(X_n, train_y)
# y_pred = ridge_model.predict(test_n)
# rmse = root_mean_squared_error(test_y, y_pred)
# logging.info('Ridge model rmse Score: %.3f' % rmse)

cv = LeaveOneOut()
scores = cross_val_score(bayesian_model, X, y_train,
                         scoring=root_mean_scorer, cv=cv, n_jobs=-1)
logging.info('Bayesian model loocv root_mean_squared_error: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# bayesian_model.fit(X_n, train_y)
# y_pred = bayesian_model.predict(test_n)
# rmse = root_mean_squared_error(test_y, y_pred)
# logging.info('Bayesian model rmse Score: %.3f' % rmse)

cv = LeaveOneOut()
scores = cross_val_score(gbr_model, X, y_train,
                         scoring=root_mean_scorer, cv=cv, n_jobs=-1)
logging.info('Gradient Boosting loocv root_mean_squared_error: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# gbr_model.fit(X_n, train_y)
# y_pred = gbr_model.predict(test_n)
# rmse = root_mean_squared_error(test_y, y_pred)
# logging.info('Gradient Boosting rmse Score: %.3f' % rmse)
