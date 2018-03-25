import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.

    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ]
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ]
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence:
                    it's size is : [ 1 x len(all_tags) ]

    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    bptrs = []
    viterbivars=[]
    # make sure end_tag exists in all_tags
    for next_tag in list(all_tags):
        value=prev_scores[0]+transition_scores[tag_to_ix[next_tag]]+Variable(torch.FloatTensor(1,len(all_tags)).fill_(cur_tag_scores[tag_to_ix[next_tag]].data[0]))
        bptrs.append(argmax(value))
        try:
            viterbivars.append(torch.max(value))
        except Exception as e:
            viterbivars.append(Variable(torch.FloatTensor(1).fill_(-np.inf)))
        '''
        for cnt,prev_tag in enumerate(list(all_tags)):
            if  to_scalar(prev_scores[0][tag_to_ix[prev_tag]])!=-np.inf and to_scalar(cur_tag_scores[tag_to_ix[next_tag]])!=-np.inf and to_scalar(transition_scores[tag_to_ix[next_tag]][tag_to_ix[prev_tag]])!=-np.inf and\
                    to_scalar(best_val<prev_scores[0][tag_to_ix[prev_tag]])+to_scalar(cur_tag_scores[tag_to_ix[next_tag]])+to_scalar(transition_scores[tag_to_ix[next_tag]][tag_to_ix[prev_tag]]):
                viterbivars[tag_to_ix[next_tag]]=prev_scores[0][tag_to_ix[prev_tag]]+cur_tag_scores[tag_to_ix[next_tag]]+transition_scores[tag_to_ix[next_tag]][tag_to_ix[prev_tag]]
                best_val=viterbivars[tag_to_ix[next_tag]]
                bptrs[tag_to_ix[next_tag]]=tag_to_ix[prev_tag]
        '''
    return viterbivars, bptrs

def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score.
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.

    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ]
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ]

    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    """

    ix_to_tag={ v:k for k,v in tag_to_ix.items() }
    #print(ix_to_tag)

    # setting all the initial score to START_TAG
    # make sure END_TAG is in all_tags
    best_path=[]
    path=[]
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32))).view(1,-1)
    path_score=0
    whole_bptrs=[]
    for m in range(len(cur_tag_scores)):
        prev_scores,bptr=viterbi_step(all_tags,tag_to_ix,cur_tag_scores[m],transition_scores,prev_scores)
        prev_scores=[x.data[0] for x in prev_scores]
        prev_scores=[prev_scores]
        prev_scores=np.asarray(prev_scores)
        prev_scores=Variable(torch.from_numpy(prev_scores.astype(np.float32))).view(1,-1)
        path.append(prev_scores)
        best_path.append(bptr)
    final_path=[]
    #print(len(cur_tag_scores))
    #print(len(path))
    #print(len(best_path))
    #print(torch.max(path[len(cur_tag_scores)-1],1)[1].data[0])
    #print(transition_scores[tag_to_ix[END_TAG]][tag_to_ix['NOUN']])
    #print(path)
    path_score+=torch.max(path[len(cur_tag_scores)-1])+transition_scores[tag_to_ix[END_TAG]][argmax(path[len(cur_tag_scores)-1])]
    #final_path.append(ix_to_tag[torch.max(path[len(cur_tag_scores)-1],1)[1].data[0]])
    final_path.append(ix_to_tag[argmax(path[len(cur_tag_scores) - 1])])
    for i in range(len(cur_tag_scores)-1,0,-1):
        #ind=torch.max(path[i-1],1)[1].data[0]
        ind=tag_to_ix[final_path[-1]]
        final_path.append(ix_to_tag[best_path[i][ind]])

    '''
    for i in range(len(cur_tag_scores)-1,-1,-1):
        sc,ind=torch.max(path[i],1)
        print(ind)
        path_score+=sc
        #print(best_path[i][ind.data[0])
        final_path.append(ix_to_tag[best_path[i][ind.data[0]]])
    '''
    best_path=final_path[::-1]
    # after you finish calculating the tags for all the words: don't forget to calculate the scores for the END_TAG
    
    
    # Calculate the best_score and also the best_path using backpointers and don't forget to reverse the path
    return path_score, best_path

    
