import operator

_UK = 'Unknown'

def basic_metrics(content:list)->dict:
    counter = {
        'total':0,
        'tp':0, 'fp': 0,
        'tn':0, 'fn': 0
    }

    for row in content:
        counter['total'] += 1

        if row['prediction'] == _UK: # classifier output is negative
            if row['truth'] == _UK:
                counter['tn'] += 1
            else:
                counter['fn'] += 1
        else:
            if row['prediction'] == row['truth']:
                counter['tp'] += 1
            else:
                counter['fp'] += 1

    assert counter['total'] == sum([counter['tp'], counter['fp'], counter['tn'], counter['fn']])
    return counter

def mismatches(content:list)->dict:
    def count_mm(k):
        if k not in mismatches.keys():
            mismatches[k] = 0
        mismatches[k] += 1

    mismatches = {} # key is a label

    for row in content:
        if row['prediction'] != _UK and row['prediction'] != row['truth']:
            count_mm(row['prediction'])

    mismatches = sorted(mismatches.items(), key=operator.itemgetter(1))
    #print(mismatches)

    return mismatches

    
"""
this code is old (does not use the prediction/truth/value dict structure)        
maybe doesnt need to
"""
def pos_neg_from_prediction(truth:list, prediction:list) -> tuple :
    """
    truth: list of classes
    prediction: list of (class, prob)
    """
    
    pos, neg = [], []
    for i in range(truth.shape[0]):
        if truth[i] == prediction[i][0]:
            pos += [float(prediction[i][1])]
        else:
            neg += [float(prediction[i][1])]
    return pos,neg

def threshold_from_fp(neg:list, target_fp:int) -> float :
    """
    neg: list of probabilities of recognitions that did not match the ground truth
    target_fp: desired absolute number of false identifications
    """

    # optim threshold to target_fp
    '''
    thresh_score = lambda t: len(filter(lambda x: x>=t, neg))

    for x in map(lambda r: r/float(1000), range(1000, 0, -1)):
        fp = thresh_score(x)
        if fp < target_fp:
            return x
    '''

    # this should do
    assert target_fp >= 1
    assert len(neg) >= 1
    return sorted(neg, reverse=True)[target_fp-1]

def dir_from_threshold(pos:list, threshold:float, N:int) -> float:
    """
    pos: list of probabilities of recognitions that matched the ground truth
    threshold: maximum probability threshold given by a fixed False Identification index (found by threshold_from_fp)
    N: total size of samples (could do this function "a menos de divisao por N")
    """

    return len(list(filter(lambda x: x>=threshold, pos))) / float(N)

def all_dirs(pos:list, neg:list) -> list:
    """
    
    """

    N = len(pos) + len(neg)
    return list(map(
        lambda n: dir_from_threshold(pos, threshold_from_fp(neg, n+1), N),
        range(len(neg)-2)
    ))
