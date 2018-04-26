import os
import csv

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
        hidden_row = row
        hidden_row.update({'truth': _UK})

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
    return relative_counter(counter)

def eval_knn(content, p_threshold:float)->dict:
    counter = {
        'total':0,
        'tp':0, 'fp': 0,
        'tn':0, 'fn': 0
    }

    for row in content:
        counter['total'] += 1

        dist = float(row['value'])
        p = dist / 0.6

        if p < p_threshold: # classifier output is negative
            if row['truth'] == _UK:
                counter['tn'] += 1
            else:
                counter['fn'] += 1
        else:
            if row['truth'] == row['prediction']:
                counter['tp'] += 1
            else:
                counter['fp'] += 1

    assert counter['total'] == sum([counter['tp'], counter['fp'], counter['tn'], counter['fn']])
    return relative_counter(counter)

def main():
    results_path = os.path.join('recognition', 'trainer', 'results')

    gallery = load_content(os.path.join(results_path, '10pp', 'gallery_test.csv'))
    hidden = hide_truth(load_content(os.path.join(results_path, '10pp', 'hidden_test.csv')))

    OSTs = [0.0, 0.01, 0.05, 0.1, 0.2]
    evm_names = list(map(lambda x: "EVM%dm" % (x*1000), OSTs))
    print(evm_names)

    hidden_content, gallery_content = {'KNN': [], 'EVM': []}, {'KNN': [], 'EVM': []}
    for i in range(len(OSTs)):
        evm_name = evm_names[i]
        #print(evm_name)

        eval_hidden = eval_evm(estimator_content(hidden, evm_name))
        eval_hidden.update({'x': OSTs[i]})
        hidden_content['EVM'].append(eval_hidden)
        
        eval_gallery = eval_evm(estimator_content(gallery, evm_name))
        eval_gallery.update({'x': OSTs[i]})
        gallery_content['EVM'].append(eval_gallery)

    many = 10
    for i in range(0, many+1):
        knn_thresh = i / many
        #print("KNN", knn_thresh)

        eval_hidden = eval_knn(estimator_content(hidden, 'KNN'), knn_thresh)
        eval_hidden.update({'x': knn_thresh})
        hidden_content['KNN'].append(eval_hidden)
        
        eval_gallery = eval_knn(estimator_content(gallery, 'KNN'), knn_thresh)
        eval_gallery.update({'x': knn_thresh})
        gallery_content['KNN'].append(eval_gallery)

    #print(gallery_content, len(gallery_content['KNN']), len(gallery_content['EVM']))
    #print(hidden_content, len(hidden_content['KNN']))

    write_content(os.path.join(results_path, '10pp', 'knn_gallery_test.csv'), gallery_content['KNN'])
    write_content(os.path.join(results_path, '10pp', 'knn_hidden_test.csv'), hidden_content['KNN'])

    write_content(os.path.join(results_path, '10pp', 'evm_gallery_test.csv'), gallery_content['EVM'])
    write_content(os.path.join(results_path, '10pp', 'evm_hidden_test.csv'), hidden_content['EVM'])

    
    #print(estimator_content(hidden, 'KNN'))



if __name__ == '__main__':
    main()
