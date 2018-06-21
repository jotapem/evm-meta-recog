import csv
import operator
import os
import sys
sys.path.append(os.getcwd())


from recognition.metrics import unconstrained_fr as ufr
_UK = ufr._UK

import argparse

def csv_parser():
    parser = argparse.ArgumentParser(description='Seeks a directory yielding prediction results as CSV files')

    parser.add_argument('results_path', metavar='P', type=str)

    return parser




def estimator_content(content:list, estim_name:str) -> list:
    """
    Filters the predictions list (content) by a estimator name
    """
    
    #return list(filter(lambda x: x['estim_name'] == estim_name, content))
    e_content = []
    for c in content:
        if c['estim_name'] == estim_name:
            if estim_name == 'KNN':
                c['value'] = '%.30f' % -float(c['value'])
            e_content.append(c)
    return e_content
            

def estimators_in_content(content:list) -> list:
    """
    Returns a list with all estimator names on the predictions list
    """
    return set(map(lambda x: x['estim_name'], content))


# I/O
def load_content(csv_path, estim_name=None):
    content = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            content.append(row)

    if estim_name:
        content = estimator_content(content, estim_name)

    return content

def write_content(csv_path, content):
    #fieldnames = ['x', 'tp', 'fp', 'tn', 'fn']
    fieldnames = content[0].keys()

    with open(csv_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in content:
            for k in row.keys():
                if type(row[k]) is float: # safe str conversion
                    row[k] = '%.20f' % row[k]

                    
            writer.writerow(row)
            #print(row)

def relative_counter(counter):
    N = float(counter['total'])
    return {
        x: counter[x] / N for x in filter(lambda k: k is not 'total', counter.keys())
    }




def main(results_path):
    #results_path = os.path.dirname(os.path.abspath(__file__))
    samples_pp = ['10pp', '20pp', '50pp']
    #samples_pp = ['10pp'] # debug



    for exp in samples_pp:
        # load respective experiment predictions from csv
        gallery = load_content(os.path.join(results_path, exp, 'gallery_test.csv'))
        hidden = ufr.hide_truth(load_content(os.path.join(results_path, exp, 'hidden_test.csv')))

        hidden_eval, gallery_eval = {}, {}
        
        # multiple estimators on the csv, so they are filtered by name
        knn_name = 'KNN'
        evm_name = sorted(list(filter(lambda x: "EVM" in x, estimators_in_content(gallery))))[-1] # currently evaluating 1 fit params variation
        print(evm_name)
        evm_name = 'EVM'


        for name in [knn_name, evm_name]:
            hidden_preds = estimator_content(hidden, name)
            gallery_preds = estimator_content(gallery, name)

            gallery_basic_metrics = ufr.basic_metrics(gallery_preds)
            print('%s %s Gallery basic metrics: %s' % (name, exp, str(gallery_basic_metrics)))

            hidden_basic_metrics = ufr.basic_metrics(hidden_preds)
            print('%s %s Hidden basic metrics: %s' % (name, exp, str(hidden_basic_metrics)))
            
            
            print('Evaluating %s %s ROC' % (name, exp))
            roc = ufr.roc_curve(hidden_preds+gallery_preds, len(gallery_preds))
            write_content(os.path.join(results_path, exp, '%s_roc.csv'%name[:3].lower()), roc)

            print('Evaluating %s %s CRR' % (name, exp))
            crr = ufr.crr_curve(hidden_preds+gallery_preds)
            write_content(os.path.join(results_path, exp, '%s_crr.csv'%name[:3].lower()), crr)



if __name__ == '__main__':
    parser = csv_parser()
    args = parser.parse_args() ; print(args)
    
    main(args.results_path)
