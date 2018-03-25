from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from gtnlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict,Counter
import torch
import torch.nn
from torch.autograd import Variable


def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    #tag1 curr_tag
    weights = defaultdict(float)
    all_tags = list(trans_counts.keys())+[END_TAG]
    for tag1 in all_tags:
        if tag1==END_TAG:
            print("hi")
        if tag1!=START_TAG:
            for tag2 in all_tags:
                if tag2!=END_TAG:
                    sum1 = sum(trans_counts[tag2].values())
                    #print(tag2, sum1,tag1)
                if tag2!=END_TAG:
                    weights[(tag1,tag2)]=np.log((trans_counts[tag2][tag1]+smoothing)/(sum1+((len(all_tags)-1)*smoothing)))

    for tag in all_tags:
        weights[(START_TAG,tag)]=-np.inf
        weights[(tag,END_TAG)]=-np.inf
    return weights

def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)
    for weight in nb_weights:
        if weight[1]!=OFFSET:
            emission_probs[word_to_ix[weight[1]]][tag_to_ix[weight[0]]]=nb_weights[weight]
    for word in vocab:
        emission_probs[word_to_ix[word]][tag_to_ix[START_TAG]]=-np.inf
        emission_probs[word_to_ix[word]][tag_to_ix[END_TAG]]=-np.inf

    for weight in hmm_trans_weights:
        tag_transition_probs[tag_to_ix[weight[0]]][tag_to_ix[weight[1]]]=hmm_trans_weights[weight]
    #for key in tag_to_ix.keys():
     #   tag_transition_probs[START_TAG][key]=-np.inf
      #  tag_transition_probs[key][END_TAG]=-np.inf
    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
    
