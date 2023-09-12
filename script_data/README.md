# Dataset

## CLEAR-Corpus

[Clear-Corpus](https://github.com/scrosseye/CLEAR-Corpus/tree/main)

### Description

It provides unique readability scores for 4724 text excerpts leveled for 3rd-12th grade readers along with information about the excerpt’s year of publishing, genre, and other meta-data.
The CLEAR corpus is meant to provide researchers interested in discourse processing and reading with a resource from which to develop and test readability metrics and to model text readability.
The CLEAR corpus includes a number of improvements in comparison to previous readability corpora including size (N = ~5,000 reading excerpts), breadth of the excerpts available, which cover over 250 years of writing in two different genres, and unique readability criterion provided for each text based on teachers’ ratings of text difficulty for student readers.
This dataset contain 4724 rows and 28 columns. 10 columns are related to the readability scores. Those columns are: BT_easiness, s.e., Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level, Automated Readability Index, SMOG Readability, New Dale-Chall Readability Formula, CAREC, CAREC_M, CML2RI.

### File Structure

| Column Name | Description |
| ----------- | ----------- |
| Label | Description |
| ID | Unique excerpt identifier |
| Author | Author of excerpt |
| Title | Title of excerpt |
| Anthology | Anthology from which excerpt was taken (if applicable) |
| URL | URL link to excerpt (if applicable) |
| Pub Year | Year of publication for excerpt |
| Categ | Category of excerpt (informative or literary) |
| Sub Cat | Sub-category for informative excerpts (if available) |
| Lexile Band | Lexile reading score band |
| Location | Location in excerpt from where except was taken |
| License | License for excerpt (if applicable) |
| MPAA Max | Motion Picture Association of America rating (from G to R) |
| MPAA #Max | Max MPAA rating given |
| MPAA# Avg | Average MPAA rating between two raters |
| Excerpt | The excerpt |
| Google WC | Word count for excerpt reported by Google |
| Sentence Count | Number of sentences in excerpt |
| Paragraphs | Number of paragraphs in excerpt |
| BT_easiness | Bradley Terry text easiness/readability score (i.e., CLEAR score) |
| s.e. | Standared error for BT_easiness score |
| Flesch-Reading-Ease | Flesch-Reading-Ease score |
| Flesch-Kincaid-Grade-Level | Flesch-Kincaid-Grade-Level score |
| Automated Readability Index | Automated Readability Index scor |
| SMOG Readability | SMOG Readability score |
| New Dale-Chall Readability Formula | New Dale-Chall Readability Formula score |
| CAREC | Crowdsourced algorithm of reading comprehension (CAREC) score |
| CAREC_M | Crowdsourced algorithm of reading comprehension (CAREC_M) score controlled for text length |
| CML2RI | Coh-Metric L2 Readability Index (CML2RI) |

## Corpus of Sentences rated with Human Complexity Judgments

[Corpus of Sentences rated with Human Complexity Judgments](http://www.italianlp.it/resources/corpus-of-sentences-rated-with-human-complexity-judgments/)

### Description

This dataset 1,200 English sentences rated by humans with a judgment of complexity. Judgments were collected through a crowdsourcing task in which 20 native speakers of each language were asked to judge how difficult they perceived a given sentence on a complexity scale from 1 (i.e. “very easy”) to 7 (i.e. “very difficult”).
The datasets of sentences used for the task were taken from the automatically converted Wall Street Journal section of the Penn Treebank for the English experiment.
This dataset contain 1200 rows and 21 columns. 20 columns are related to the judgement scores by the human.

### File Structure

The data is present in csv format. The columns are as follows:

- ID: unique identifier of the sentence
- Sentence: the sentence
- 20 columns: the judgments of the 20 annotators

### Citation

```[bibtex]
Brunato D., De Mattei L., Dell’Orletta F., Iavarone B., Venturi G. (2018) “Is this Sentence Difficult? Do you Agree?“. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), 31–4 November, Bruxelles.
```

## REALEC

[REALEC](https://realec.org)

### Description

REALEC, learner corpus released in the open access, have 6,054 essays written in English by HSE undergraduate students in their English university-level examination by the year 2020.
Lots of these essays has been given CEFR level based on proficiency. This [paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-16270-1_7) provides methods used to collect and annotate the data.
Currently, we have taken EXAM 2016 folder of dataset which had 1334 essays out of which only 538 entries had CEFR scored with it. We have taken only those entries which had CEFR score with it. This can be extended to other dataset present in REALEC.

### File Structure

The data is generatd in csv format. The columns are as follows:

- sentence:  Text for the complexity
- ielts: Is it part of the Ielts exam
- CEFR_level: CEFR level of the text
- work_type: From where text has been taken, for example exam, essay, etc.
- year: Year of the exam
