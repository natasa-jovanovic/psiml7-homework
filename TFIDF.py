from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import os
import operator
import math
import time


def path_to_files(corpus_name):
    corpus = []
    directory = corpus_name
    for name in os.listdir(directory):
        if name.endswith(".txt"):
            corpus.append(directory + '/' + name)
        else:
            corpus.extend(path_to_files(directory + '/' + name))
    return corpus


def list_terms(file):
    return reduce(word_tokenize(file))


def reduce(words):
    words_reduced = []
    for word in words:
        if word.isalnum(): words_reduced.append(stemmer.stem(word))
    return words_reduced


def termfreq(list_terms):
    term_freq = {}
    for term in list_terms:
        if term in term_freq:
            term_freq[term] += 1
        else:
            term_freq[term] = 1
    return dict(sorted(term_freq.items(), key=operator.itemgetter(1), reverse=True))


def tfidf_score(input_terms, term_corpus):
    term_score = termfreq(input_terms)
    N = len(documents_paths)
    for term in term_score:
        count = 0
        for terms in term_corpus:
            if term in terms:
                count += 1
        term_score[term] *= math.log(N / count)
    term_score = sorted(term_score.items(), key=lambda x: (-x[1], x[0]))
    term_score = {k: v for k, v in term_score}
    return term_score


def sentences_summary(sentences, tf_idf_score):
    sentences_summary = {}
    for sentence in sentences:
        terms_per_sentence = reduce(word_tokenize(sentence))
        sentence_scores = []
        for term in terms_per_sentence:
            sentence_scores.append([term, tf_idf_score[term]])

        sentence_scores = sorted(sentence_scores, key=lambda x: (-x[1], x[0]))
        if len(sentence_scores) > 10: sentence_scores = sentence_scores[0:10]
        # print(sentence, sentence_scores)

        for sent_term in sentence_scores:
            if sentence in sentences_summary:
                sentences_summary[sentence] += sent_term[1]
            else:
                sentences_summary[sentence] = sent_term[1]
    temp_summary = sorted(sentences_summary.items(), key=lambda x: (-x[1]))
    temp_summary = {k: v for k, v in temp_summary}
    temp_summary = firstN(temp_summary, 5)

    count = 1
    summary = ""
    for sent in sentences_summary:
        if [sent, sentences_summary[sent]] in temp_summary:
            summary += sent + ' '
            count += 1
        if count == 6:
            break

    print(summary[:-1])



def print_top_terms(tf_idf_score):
    top_terms = firstN(tf_idf_score, 10)
    output = ""
    for term in top_terms:
        output += term[0] + ', '
    print(output[:-2])


def firstN(tf_idf_score, N):
    temp = []
    count = 1
    for item in tf_idf_score:
        if count <= N:
            temp.append([item, tf_idf_score[item]])
        else:
            break
        count += 1
    return temp



stemmer = SnowballStemmer("english")

start_time = time.time()
documents_paths = path_to_files('corpus')
term_corpus = []
for doc in documents_paths:
    doc = open(doc, 'r', encoding='UTF8')
    term_corpus.append(list_terms(doc.read()))
inp = open('corpus\Scotland\Scotland.txt', 'r', encoding='UTF8')
input_doc = inp.read()
inp.close()

tf_idf_score = tfidf_score(list_terms(input_doc), term_corpus)
print_top_terms(tf_idf_score)

sentences_summary(sent_tokenize(input_doc), tf_idf_score)

print("time elapsed: {:.2f}s".format(time.time() - start_time))