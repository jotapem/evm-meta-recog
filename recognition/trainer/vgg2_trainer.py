import copy
import csv
import time
import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import pickle

from sklearn.metrics import make_scorer, accuracy_score
from sklearn import metrics as skmetrics
from sklearn.model_selection import GridSearchCV
import torch
#import torchvision

from recognition.dataset.vgg_face2 import vggDataset
from recognition.estimator.evm import EVM
from recognition.estimator.knn import KNN
from recognition.metrics.unconstrained_fr import all_dirs, pos_neg_from_prediction

def print_dict(d):
    for k in sorted(d.keys(),reverse=True):
        v = d[k]
    #for k,v in d.items():
        print("[%s] \n %s \n" % (k,v))

def measure_call(call, n_samples=1000, verbose=False):
    times = []
    for i in range(n_samples):
        start = time.time()
        call()
        end = time.time()
        times += [1000 * (end - start)] # to ms

    if verbose:
        print(times)
    
    times = times[1:]
    return np.mean(times), np.var(times)

def pickle_load(path):
    with open(path, 'rb') as pickle_in:
        return pickle.load(pickle_in)

def get_classifier_data(data):
    '''
    X = np.array(list(map(lambda x: pickle_load(x['descriptor_path']), data)))
    Y = np.array(list(map(lambda x: np.array(x['label']), data)))
    '''
    X, Y = data
    return X, Y


# train test 
def train(estim, data, verbose=False):
    X, Y = get_classifier_data(data)
    if verbose: print('Train shapes', X.shape, Y.shape)

    estim.fit(X, Y)

    if verbose: print('Finished training')

def test_logging(estims:dict, data, log_out:str):
    X, Y = get_classifier_data(data)
    print('Test shapes', X.shape, Y.shape)

    csv_content = []
    for estim_name, estim in estims.items():
        print("Logging %s" % estim_name)
        
        Y_ = estim.predict_with_prop(X)
        #print(p)

        for i in range(len(Y_)):
            pred, value = Y_[i]
            truth = Y[i]
            
            line = {'estim_name': estim_name, 'prediction': pred, 'truth': truth, 'value': value}
            #print(line)
            csv_content += [line]

    with open(os.path.join('recognition', 'trainer', 'results', log_out), 'w') as csv_file:
        field_names = ['estim_name', 'prediction', 'truth', 'value']
        writer = csv.DictWriter(csv_file, fieldnames=field_names)

        writer.writeheader()
        for line in csv_content:
            writer.writerow(line)


# model io
def load_model(name):
    return pickle_load(os.path.join('models', name))

def save_model(name, model):
    with open(os.path.join('models', name), 'wb') as pickle_out:
        pickle.dump(model, pickle_out, pickle.HIGHEST_PROTOCOL)
    print('Saved model ', name)

def expand_evm(estim, param_ranges) -> dict:
    """
    takes a trained EVM and returns a dict changing its open_set_threshold param
    estimator dict ::= {estimator_name:str, estimator}
    """
    e = {}

    for o in param_ranges:
        _ = copy.copy(estim)
        _.open_set_threshold = o
        e["EVM%dm" % (1000*o)] = _

    return e


# the gold (o grosso)
def experiment(samples_per_person:int, persons:int):
    hidden_dataset = vggDataset('/media/Datasets/vgg_face2', 'test')#, verbose=True)
    gallery_dataset = vggDataset('/media/Datasets/vgg_face2', 'train_aligned')
    
    train_data, test_data = gallery_dataset.get_training_data(samples_per_person, samples_test=5, persons_limit=persons) # this gives persons * 5 gallery test samples

    
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


    estimators = { # this is a dict estimator like a data type (e.g return of expand_evm())
        'EVM': EVM(use_gpu=True, open_set_threshold=0.0),
        'KNN': KNN(1)
    }

    
    reload_models=False # hmmmm i dont think i can get always the sample train/test split so lets not reload
    

    if reload_models:
        print("Reloading pretrained models")

        for name, estim in estimators.items():
            estimators[name] = load_model(name+'.pkl')

    # plain train
    else:
        print("Training raw models")

        for name, estim in estimators.items():
            
            train(estim, train_data, verbose=True)
            save_model(name+'.pkl', estim)

    '''
    # dataloader train
    for i, data in enumerate(dataloader, 0):
        X,Y = data
        print(X, Y)

        evm.fit(X.numpy(),Y.numpy())
        print('[%d] fit' % (i))
    '''

    # logging test
    OSTs = [0.0, 0.001, 0.005, 0.01, 0.02]
    test_estimators = expand_evm(estimators['EVM'], OSTs)
    test_estimators.update({'KNN': estimators['KNN']})

    test_logging(test_estimators, test_data, os.path.join("%dpp" % samples_per_person, 'gallery_test.csv'))
    hidden_data, _ = hidden_dataset.get_training_data(20) # this gives 500*20 = 10k hidden samples
    test_logging(test_estimators, hidden_data, os.path.join("%dpp" % samples_per_person, 'hidden_test.csv'))


def performance_eval(estim, fit_data, predict_data, n_samples:int, verbose=True):
    def vprint(s):
        if verbose:
            print(s)
    
    

    #estim = EVM(open_set_threshold=0.01)

    fit_x, fit_y = fit_data
    vprint("doing a fit")
    fit_t = measure_call(lambda: estim.fit(fit_x, fit_y), verbose=True, n_samples=5)#n_samples)
    vprint(fit_t)

    
    pred_x, _ = predict_data    
    vprint("doing a predict")
    pred_t = measure_call(lambda: estim.predict(pred_x), n_samples=n_samples)
    vprint(pred_t)

    return {'fit': fit_t, 'pred': pred_t}

def performance_multieval(estims:dict, samples_per_person:int, persons:int, n_samples:int, verbose=False):
    #hidden_dataset = vggDataset('/media/Datasets/vgg_face2', 'test')#, verbose=True)
    gallery_dataset = vggDataset('/media/Datasets/vgg_face2', 'train_aligned')

    fit_data, _ = gallery_dataset.get_training_data(samples_per_person, 0, persons)
    predict_data, _ = gallery_dataset.get_training_data(1, 0, 1)
    
    performance = {}
    for k in sorted(models.keys()):
        print("\nEvaluating %s with %d persons and %d samples-per-person" % (k, persons, samples_per_person))
        performance.update({k: performance_eval(models[k], fit_data, predict_data, n_samples=100, verbose=verbose)})

    print_dict(performance)
    
    
if __name__ == '__main__':
    #models = {("EVM_Red_%4d" % int(r)): EVM(redundancy_rate=r/1000.) for r in range(0, 1000, 25)}
    models = {("EVM_BiasD_%4d" % int(r)): EVM(redundancy_rate=r/1000., biased_distance=r/1000.) for r in range(400, 1000, 50)}
    models.update({"KNN": KNN(1)})
    #print(models)

    performance_multieval(models, 20, 1500, n_samples=100, verbose=True)
    '''
    performance = {
        'KNN': performance_eval(load_model('KNN.pkl'),n_samples=100),
        'EVM': performance_eval(load_model('EVM.pkl'),n_samples=100)
    }
    print(performance)
    '''
    '''
    for pp in [10, 20, 50]:
        experiment(pp, 8000)
    '''
    
    

    '''
    # direct test
    test_evm(evm, test_data, estim_name='EVM', dataset_name='Known people', verbose=True)
    test_knn(knn, test_data, estim_name='KNN', dataset_name='Known people')
    test_evm(evm, hidden_dataset.data, estim_name='EVM', dataset_name='New people')
    test_knn(knn, hidden_dataset.data, estim_name='KNN', dataset_name='New people')

    # grid stuff
    evm_params = {
        'open_set_threshold': list(map(lambda x: x/float(1000), range(1, 1000, 100))),
        'tail': range(10, 1000, 10)
    }
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        #'precision': 'precision'
    }

    evm_grid = GridSearchCV(evm, param_grid=evm_params, scoring=scorers, n_jobs=-1, refit='accuracy')
    
    train(evm_grid, dataset.train_data, verbose=True)

    test(evm_grid, dataset.test_data, estim_name='EVM_grid', dataset_name='Known people')
    test(evm_grid, (hidden_dataset.train_data + hidden_dataset.test_data), estim_name='EVM_grid', dataset_name='New people')
    '''

    
    
    
"""
def test_evm(estim, data, verbose=False, estim_name='Estimator', dataset_name='Dataset'):
    X, Y = get_classifier_data(data)
    Y_ = estim.predict(X)
    if verbose: print('Test shapes', X.shape, Y.shape, Y_.shape)

    if isinstance(estim, GridSearchCV):
        print("Best params: %s" % (
            estim.best_params_
        ))
    
    total = Y.shape[0]
    correct = (Y_ == Y).sum()
    unknowns = (Y_ == 'Unknown').sum()
    mismatches = total - (correct + unknowns)

    #precision = skmetrics.precision_score(Y, Y_)
    #recall = skmetrics.recall_score(Y, Y_)
    
    print("[%s/%s] Properly recognized: %f%% ; Unknowns/False-Negatives: %f%% ; False-Positives/Mismatches: %f%% ; Samples: %d\n" % (
        estim_name, dataset_name,
        100*correct/float(total),
        100*unknowns/float(total),
        100*mismatches/float(total),
        total
    ))

    return {
        'total': total,
        'correct': correct,
        'unknowns': unknowns,
        'mismatches': mismatches
    }

def test_knn(estim, data, k_thresh=0.6, verbose=False, estim_name='KNN', dataset_name='Dataset'):
    X, Y = get_classifier_data(data)
    Y_ = estim.predict(X)
    Y_k = estim.kneighbors(X)

    total = Y.shape[0]
    correct, unknowns, mismatches = 0, 0, 0
    for i in range(total):
        if Y_k[0][i] > k_thresh:
            unknowns += 1
        else:
            if Y_[i] == Y[i]:
                correct += 1
            else:
                mismatches += 1
        
    print("[%s/%s] Properly recognized: %f%% ; Unknowns/False-Negatives: %f%% ; False-Positives/Mismatches: %f%% ; Samples: %d\n" % (
        estim_name, dataset_name,
        100*correct/float(total),
        100*unknowns/float(total),
        100*mismatches/float(total),
        total
    ))

    return {
        'total': total,
        'correct': correct,
        'unknowns': unknowns,
        'mismatches': mismatches
    }
      
        

def DIR_test(estim, data, verbose=False, estim_name='Estimator', dataset_name='Dataset'):
    X, Y = get_classifier_data(data)
    Y_ = estim.predict_proba(X)

    #print("Achou ", Y_, len(Y_))
    
    pos, neg = pos_neg_from_prediction(Y, Y_)
    dirs = all_dirs(pos, neg)

    print(min(neg), min(pos))
    print("[%s/%s] DIRs: %s" % (
        estim_name, dataset_name,
        dirs
    ))
    
    return dirs
"""
