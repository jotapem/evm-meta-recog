import operator

"""
content refers as list of dicts (prediction rows) with fields
 truth : string
 prediction : string
 value : float
other fields are not seen by this module
"""

_UK = 'Unknown'

spos_filter = lambda x: x['truth'] != _UK and x['truth'] == x['prediction']
sneg_filter = lambda x: not spos_filter(x)
snegneg_filter = lambda x: sneg_filter(x) and x['truth'] == _UK

def range_sample(r:list, how_much:int):
    r_len = len(r)
    how_much = r_len if r_len < how_much else int(how_much)
    return map(lambda x: r[int(r_len*x/how_much)], range(how_much))
    

def hide_truth(content:list)->list:
    """
    Force that the ground truth for predictions in content is a unknown constant _UK
    Useful when dealing with a open-set scenario (when a recognition query might yield a negative prediction)
    """
    
    hidden_content = []

    for row in content:
        hidden_row = dict(row)
        hidden_row.update({'truth': _UK})

        hidden_content.append(hidden_row)

    return hidden_content

def threshold_pred(content, thresh:float, logic='restrictive'):
    """
    Uses a threshold logic on the prediction value (similarity score) to yield negative predictions
    logic parameter should be a function!!!!

    Useful as a generic way to model a 'unkown awareness' based on a similarity/confidence score on the prediction
    """
    
    assert logic in ['restrictive', 'permissive']

    hidden_content = []

    for row in content:
        hidden_row = dict(row)
        v = float(row['value'])
        if ((logic is 'restrictive' and v > thresh) or (logic is 'permissive' and v < thresh)):
            hidden_row.update({'prediction': _UK})

        hidden_content.append(hidden_row)

    return hidden_content


def basic_metrics(content:list)->dict:
    """
    Returns a dictionary counting true/false positive/negatives as it iterates over the prediction list 
    The keys looked up on content are only 'truth' and 'prediction'
    Functions similar to threshold_pred and hide_truth should be used in order to evaluate true/positive negatives

    True Positives: Query of enrolled person X Predicted the same (enrolled) person
    True Negative: Query of enrolled person X Predicted as unknown
    False Positive: Query of enrolled person X Predicted as any other enrolled person
    False Negative: Query of enrolled person X Predicted as unknown
    """
    
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
    """
    Histogram-like count of how many mismatches a label was responsible for
    """
    
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

def roc_curve(content:list, N:int) -> dict :
    """
    Evaluates a ROC (TP-vs-FP) curve (there should be the rejection ROC curve as well)
    FP values yield a threshold that is applied to compute TP

    N:int is the total number of enrolled predictions to be made. Could be computed from content under the assumptions that 
      1) predictions from unknowns are already hidden 
      2) every desired enrolled identification is in the prediction list
    """
    # splits the prediction list, mapping each row to its similarity score as a float
    spos = list(map(lambda x: float(x['value']), filter(spos_filter, content)))
    sneg = list(map(lambda x: float(x['value']), filter(sneg_filter, content)))

    # uses S- to generate thresholds
    sneg.sort(reverse=True)
    rate_range = list(sorted(map(lambda t: t, set(sneg)),reverse=False))
    rate_range_shorter = range_sample(rate_range, 1e2)


    roc_content = []
    #for t in rate_range:
    for t in rate_range_shorter:
        # clear threshold apply
        fr = len(list(filter(lambda x: x>=t, sneg))) / float(len(sneg)) 
        tr = len(list(filter(lambda x: x>=t, spos))) / float(N)

        roc_content.append({
            'threshold': t,
            'x': fr,
            'y': tr
        })

    return roc_content

def crr_curve(content:list) -> dict:
    """
    Evaluates a Correct Rejection Curve (similar to a ROC curve)
    TP values yield a threshold that is applied to compute TN (correct rejections)

    The set of samples-to-be-rejected (snegneg) is computed from content under no assumptions
    """

    spos = list(map(lambda x: float(x['value']), filter(spos_filter, content)))
    snegneg = list(map(lambda x: float(x['value']), filter(snegneg_filter, content)))

    # uses S+ to generate thresholds
    spos.sort()
    rate_range = list(sorted(map(lambda t: t, set(spos))))
    rate_range_shorter = range_sample(rate_range, 1e2)

    crr_content = []
    for t in rate_range_shorter:
        tr = len(list(filter(lambda x: x>=t, spos))) / float(len(spos))
        fr = len(list(filter(lambda x: x<t, snegneg))) / float(len(snegneg))

        crr_content.append({
            'threshold': t,
            'y': fr,
            'x': tr
        })

    return crr_content

