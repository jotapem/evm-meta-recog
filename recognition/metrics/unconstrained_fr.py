# "1708.02337" metric
def pos_neg_from_prediction(truth:list, prediction:list) -> tuple :
    """
    truth: list of classes
    prediction: list of (class, prob)
    """
    
    pos, neg = [], []
    for i in truth.shape[0]:
        if truth[i] == prediction[i][0]:
            pos += [prediction[i][1]]
        else:
            neg += [prediction[i][1]]
    return pos,neg

def threshold_from_fp(neg:list, target_fp:int) -> float :
    """
    neg: list of probabilities of recognitions that did not match the ground truth
    target_fp: desired absolute number of false identifications
    """

    # optim threshold to target_fp
    thresh_score = lambda t: len(filter(lambda x: x>=t, neg))
    #thresh, fp = 1, thresh_score(1)

    # please dont flame me
    for x in map(lambda r: r/float(1000), range(1000, 0, -1)):
        fp = thresh_score(x)
        if fp < target_fp:
            return x

def dir_from_threshold(pos:list, threshold:float, N:int) -> float:
    """
    pos: list of probabilities of recognitions that matched the ground truth
    threshold: maximum probability threshold given by a fixed False Identification index (found by threshold_from_fp)
    N: total size of samples (could do this function "a menos de divisao por N")
    """

    return len(filter(lambda x: x>=threshold, pos)) / float(N)
