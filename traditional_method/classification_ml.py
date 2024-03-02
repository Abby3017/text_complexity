import logging
from collections import Counter
from math import log

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import scipy
import sklearn
from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, KFold, LeaveOneOut,
                                     RepeatedKFold, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, filename='classification_ml_with_diff_rmse_syn.log',
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
merge_df = merge_df[['cefr_numeric_x', 'cefr_numeric_syn_x', 'sentences_x', 'syn_aug_x', 'word_freq_sentence',
                     'word_freq_syn', 'freq_diff_syn', 'bert_score_syn', 'complexity_score', 'complexity_score_syn']]

merge_df['syn_cefr_change'] = abs(
    merge_df['cefr_numeric_syn_x'] - merge_df['cefr_numeric_x'])
merge_df['syn_complexity_change'] = abs(
    merge_df['complexity_score_syn'] - merge_df['complexity_score'])

merge_df = merge_df[['sentences_x', 'syn_aug_x', 'syn_complexity_change', 'cefr_numeric_syn_x',
                     'syn_cefr_change', 'bert_score_syn', 'freq_diff_syn', 'cefr_numeric_x']]

# get more attributes from sentences
# get number of words using nltk tokenizer
merge_df['num_words'] = merge_df['syn_aug_x'].apply(
    lambda x: len(nltk.word_tokenize(x)))
# get average word length using nltk tokenizer
merge_df['avg_word_length'] = merge_df['syn_aug_x'].apply(
    lambda x: np.mean([len(word) for word in nltk.word_tokenize(x)]))
# count number of syllables in sentences
merge_df['num_syllables'] = merge_df['syn_aug_x'].apply(
    lambda x: syllable_count(x))
# get number of unique tokens in sentences
merge_df['num_unique_tokens'] = merge_df['syn_aug_x'].apply(
    lambda x: uniq_tokens(x))
# get type token ratio
merge_df['ttr'] = merge_df['syn_aug_x'].apply(lambda x: calculate_ttr(x))
# get lexical density
merge_df['lexical_density'] = merge_df['syn_aug_x'].apply(
    lambda x: calculate_density(x))
# get uber index
merge_df['uber_index'] = merge_df['syn_aug_x'].apply(
    lambda x: calculate_uber_index(x))
# average word length in syllables
merge_df['avg_word_length_syllables'] = merge_df['num_syllables'] / \
    merge_df['num_words']
# calculate_pos return dictionary with noun, adjective, verb, adverb counts in a sentence move them to separate columns for each sentence
merge_df[['noun_count', 'adjective_count', 'verb_count', 'adverb_count']] = pd.DataFrame(
    merge_df['syn_aug_x'].apply(lambda x: calculate_pos(x)).tolist(), index=merge_df.index)

y_train_change = merge_df['syn_cefr_change']
y_cefr = merge_df['cefr_numeric_syn_x']
x_cefr_values = merge_df['cefr_numeric_x']
X_train = merge_df.drop(
    ['syn_aug_x', 'syn_cefr_change', 'sentences_x', 'cefr_numeric_syn_x'], axis=1)

logging.info(X_train.columns.tolist())

scaler = StandardScaler()
scaler.fit(X_train)
X = scaler.transform(X_train)

cv = LeaveOneOut()
lr_predictions = []
dt_predictions = []
rf_predictions = []
g_predictions = []
cefr_values = []

lr_model = LogisticRegression(
    penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=0.5)
dt_model = DecisionTreeClassifier(
    max_depth=3, class_weight="balanced", random_state=1)
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, class_weight="balanced", random_state=1)
g_model = GaussianNB()

for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_train_change[train_index], y_train_change[test_index]
    y_current_cefr = y_cefr[test_index]
    x_current_cefr = x_cefr_values[test_index]
    cefr_values.append(y_current_cefr.values[0])
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    cefr_predicted = x_current_cefr.values[0] + y_pred[0]
    lr_predictions.append(cefr_predicted)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    cefr_predicted = x_current_cefr.values[0] + y_pred[0]
    dt_predictions.append(cefr_predicted)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    cefr_predicted = x_current_cefr.values[0] + y_pred[0]
    rf_predictions.append(cefr_predicted)
    g_model.fit(X_train, y_train)
    y_pred = g_model.predict(X_test)
    cefr_predicted = x_current_cefr.values[0] + y_pred[0]
    g_predictions.append(cefr_predicted)

lr_predictions = np.array(lr_predictions)
dt_predictions = np.array(dt_predictions)
rf_predictions = np.array(rf_predictions)
g_predictions = np.array(g_predictions)
cefr_values = np.array(cefr_values)

rmse = root_mean_squared_error(cefr_values, lr_predictions)
logging.info('Logistic Regression rmse Score: %.3f' % rmse)
rmse = root_mean_squared_error(cefr_values, dt_predictions)
logging.info('Decision Tree rmse Score: %.3f' % rmse)
rmse = root_mean_squared_error(cefr_values, rf_predictions)
logging.info('Random Forest rmse Score: %.3f' % rmse)
rmse = root_mean_squared_error(cefr_values, g_predictions)
logging.info('Gaussian NB rmse Score: %.3f' % rmse)

lr_model = LogisticRegression(
    penalty="elasticnet", class_weight="balanced", solver="saga", l1_ratio=0.5)
dt_model = DecisionTreeClassifier(
    max_depth=3, class_weight="balanced", random_state=1)
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, class_weight="balanced", random_state=1)
g_model = GaussianNB()

# cv = LeaveOneOut()
scores = cross_val_score(lr_model, X, y_train_change,
                         scoring='accuracy', cv=cv, n_jobs=-1)
logging.info('Logistic Regression loocv Accuracy: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# lr_model.fit(X, y_train)
# y_pred = lr_model.predict(X)
# prediction_score = lr_model.score(X, y_train)
# logging.info('Logistic Regression Prediction Score: %.3f' % prediction_score)

# cv = LeaveOneOut()
scores = cross_val_score(dt_model, X, y_train_change,
                         scoring='accuracy', cv=cv, n_jobs=-1)
logging.info('Decision Tree loocv Accuracy: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# dt_model.fit(X, y_train)
# y_pred = dt_model.predict(X)
# prediction_score = dt_model.score(X, y_train)
# logging.info('Decision Tree Prediction Score: %.3f' % prediction_score)

# cv = LeaveOneOut()
scores = cross_val_score(rf_model, X, y_train_change,
                         scoring='accuracy', cv=cv, n_jobs=-1)
logging.info('Random Forest loocv Accuracy: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# rf_model.fit(X, y_train)
# y_pred = rf_model.predict(X)
# prediction_score = rf_model.score(X, y_train)
# logging.info('Random Forest Prediction Score: %.3f' % prediction_score)

# cv = LeaveOneOut()
scores = cross_val_score(g_model, X, y_train_change,
                         scoring='accuracy', cv=cv, n_jobs=-1)
logging.info('Gaussian NB loocv Accuracy: %.3f (%.3f)' %
             (mean(scores), std(scores)))
# g_model.fit(X, y_train)
# y_pred = g_model.predict(X)
# prediction_score = g_model.score(X, y_train)
# logging.info('Gaussian NB Prediction Score: %.3f' % prediction_score)
