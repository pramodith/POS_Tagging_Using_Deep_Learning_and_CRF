from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation, preproc

import numpy as np
from collections import defaultdict,Counter

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preproc.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights


def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts=Counter()
    for cnt,corpus in enumerate(x):
        if y[cnt]==label:
            corpus_counts.update(corpus)
    c= defaultdict(float,dict(corpus_counts))
    for key in c:
        c[key]=float(c[key])
    return c
    


def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    den = 0.0
    lprobs = defaultdict(float)
    corpus_counts = get_corpus_counts(x, y, label)
    den = sum(corpus_counts.values())
    den = den + len(vocab) * smoothing
    for word in vocab:
        if word in corpus_counts:
            lprobs[word] = np.log((corpus_counts[word] + smoothing) / den)
        else:
            lprobs[word] = np.log(smoothing / den)
    return lprobs
    


def estimate_nb(x,y,smoothing):
    """
    estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    vocab = []
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
    total=len(y)
    for inst in x:
        vocab.extend(list(inst.keys()))
    vocab = set(vocab)
    labels = set(y)
    for label in labels:
        phi = estimate_pxy(x, y, label, smoothing, vocab)
        try:
            #print(y,label)
             phi[OFFSET] = np.log(y.count(label) / float(total))
        except Exception as e:
            print(e)
        for key in phi:
            try:
                doc_counts[(label, key)] = phi[key]
            except Exception as e:
                pass
                #print(e)
                #print(float(total))
                #print(np.count_nonzero(y == label) / float(total))
    # print (doc_counts)
    return doc_counts
    


def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    score = {}
    for smoother in smoothers:
        theta_nb = estimate_nb(x_tr, y_tr, smoother)
        y_hat = clf_base.predict_all(x_dv, theta_nb, set(y_tr))
        score[smoother] = (evaluation.acc(y_hat, y_dv))
    return clf_base.argmax(score), score







