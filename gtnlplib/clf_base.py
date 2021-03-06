from gtnlplib.constants import OFFSET

import operator

# use this to find the highest-scoring label
argmax = lambda x : max(x.items(),key=operator.itemgetter(1))[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    raise NotImplementedError
    
    

def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    labels = list(labels)
    score = {label: 0.0 for label in labels}
    base_features[OFFSET] = 1
    # if len(weights.keys())>0:
    for weight in weights.keys():
        #print(weight)
        score[weight[0]] += base_features[weight[1]] * weights[weight]
    #         if weight[1]==OFFSET:
    #             ans[labels.index(weight[0])]+=1*weights[weight]
    return argmax(score), score
