import csv
import operator
import os
import sys
sys.path.append(os.getcwd())


from recognition.metrics import unconstrained_fr

_UK = 'Unknown'


# io
def estimator_content(content, estim_name):
    return list(filter(lambda x: x['estim_name'] == estim_name, content))

def estimators_in_content(content):
    return set(map(lambda x: x['estim_name'], content))

def load_content(csv_path, estim_name=None):
    content = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            content.append(row)

    if estim_name:
        content = estimator_content(content, estim_name)

    #estimators = estimators_in_content(content)
    #print("Loaded csv at %s containing the estimators %s" % (csv_path, estimators))

    return content

def write_content(csv_path, content):
    fieldnames = ['x', 'tp', 'fp', 'tn', 'fn']

    with open(csv_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in content:
            writer.writerow(row)
            #print(row)

def hide_truth(content):
    hidden_content = []

    for row in content:
        hidden_row = dict(row)
        hidden_row.update({'truth': _UK})

        hidden_content.append(hidden_row)

    return hidden_content

def hide_pred(content, thresh, logic='restrictive'):
    assert logic in ['restrictive', 'permissive']

    hidden_content = []

    for row in content:
        hidden_row = dict(row)
        v = float(row['value'])
        if ((logic is 'restrictive' and v > thresh) or (logic is 'permissive' and v < thresh)):
            hidden_row.update({'prediction': _UK})

        hidden_content.append(hidden_row)

    return hidden_content

def relative_counter(counter):
    N = float(counter['total'])
    return {
        x: counter[x] / N for x in filter(lambda k: k is not 'total', counter.keys())
    }

# estimator specific eval

def eval_evm(content)->dict:
    counter = {
        'total':0,
        'tp':0, 'fp': 0,
        'tn':0, 'fn': 0
    }
    mismatches = {} # key is a class
    def count_mm(k):
        if k not in mismatches.keys():
            mismatches[k] = 0
        mismatches[k] += 1

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
                count_mm(row['prediction'])

    mismatches = sorted(mismatches.items(), key=operator.itemgetter(1))
    #print(mismatches)

    assert counter['total'] == sum([counter['tp'], counter['fp'], counter['tn'], counter['fn']])
    return relative_counter(counter)

def eval_knn(content, p_threshold:float)->dict:
    '''
    counter = {
        'total':0,
        'tp':0, 'fp': 0,
        'tn':0, 'fn': 0
    }

    for row in content:
        counter['total'] += 1

        dist = float(row['value'])
        p = dist #/ 0.6

        if p > p_threshold: # classifier output is negative
            if row['truth'] == _UK:
                counter['tn'] += 1
            else:
                counter['fn'] += 1
        else:
            if row['truth'] == row['prediction']:
                counter['tp'] += 1
            else:
                counter['fp'] += 1
    '''
    h_content = hide_pred(content, p_threshold)
    g_eval = eval_evm(h_content)
    #print(p_threshold, g_eval)
    return g_eval
    
    assert counter['total'] == sum([counter['tp'], counter['fp'], counter['tn'], counter['fn']])
    return relative_counter(counter)

def main():
    results_path = os.path.join('recognition', 'trainer', 'results')
    samples_pp = ['10pp']#, '20pp', '50pp']

    for exp in samples_pp:
        gallery = load_content(os.path.join(results_path, exp, 'gallery_test.csv'))
        hidden = hide_truth(load_content(os.path.join(results_path, '10pp', 'hidden_test.csv')))

        evm_names = list(filter(lambda x: "EVM" in x, estimators_in_content(gallery)))
        print(evm_names)

        hidden_content, gallery_content = {'KNN': [], 'EVM': []}, {'KNN': [], 'EVM': []}
        for i in range(len(evm_names)):
            evm_name = evm_names[i]
            ost = int(evm_name[3:-1]) / 1000.
            #print(evm_name, ost)

            eval_hidden = eval_evm(estimator_content(hidden, evm_name))
            eval_hidden.update({'x': ost})
            hidden_content['EVM'].append(eval_hidden)

            eval_gallery = eval_evm(estimator_content(gallery, evm_name))
            eval_gallery.update({'x': ost})
            gallery_content['EVM'].append(eval_gallery)

        many = 10
        for i in range(0, many+1):
            knn_thresh = i / float(many)
            #print("KNN", knn_thresh)

            eval_hidden = eval_knn(estimator_content(hidden, 'KNN'), knn_thresh)
            eval_hidden.update({'x': knn_thresh})
            hidden_content['KNN'].append(eval_hidden)

            eval_gallery = eval_knn(estimator_content(gallery, 'KNN'), knn_thresh)
            eval_gallery.update({'x': knn_thresh})
            gallery_content['KNN'].append(eval_gallery)

        #print(gallery_content, len(gallery_content['KNN']), len(gallery_content['EVM']))
        #print(hidden_content, len(hidden_content['KNN']))

        write_content(os.path.join(results_path, exp, 'knn_gallery_test.csv'), gallery_content['KNN'])
        write_content(os.path.join(results_path, exp, 'knn_hidden_test.csv'), hidden_content['KNN'])

        write_content(os.path.join(results_path, exp, 'evm_gallery_test.csv'), gallery_content['EVM'])
        write_content(os.path.join(results_path, exp, 'evm_hidden_test.csv'), hidden_content['EVM'])


        #print(estimator_content(hidden, 'KNN'))



if __name__ == '__main__':
    main()
