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
#from recognition.dataset.vgg_arcface import vggDataset
from classifiers.evm import EVM
from classifiers.knn import KNN
#from recognition.metrics.unconstrained_fr import all_dirs, pos_neg_from_prediction

def print_dict(d):
    for k in sorted(d.keys(),reverse=True):
        v = d[k]
    #for k,v in d.items():
        print("[%s] \n %s \n" % (k,v))

def vprint(s, v):
    if v:
        print(s)

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

        for i in range(len(Y_)):
            def format_value(f):
                #print('trying to format %s of type %s' % (f, type(f)))
                return '%.10f' % float(f)
            
            pred, value = Y_[i]
            print(pred, value, Y_[i])
            truth = Y[i]
            
            line = {
                'estim_name': estim_name,
                'prediction': pred,
                'truth': truth,
                'value': format_value(value)
            }
            #print(line)
            csv_content += [line]

    with open(os.path.join('trainer', 'results', log_out), 'w') as csv_file:
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
def experiment(samples_per_person:int, persons:int, verbose=False):
    hidden_dataset = vggDataset('/media/Datasets/vgg_face2', 'test', verbose=verbose)
    gallery_dataset = vggDataset('/media/Datasets/vgg_face2', 'train_aligned', verbose=verbose)

    hidden_data, _ = hidden_dataset.get_training_data(20) # this gives 500*20 = 10k hidden samples
    #hidden_data, _ = hidden_dataset.get_training_data(1) # useful for debugging
    train_data, test_data = gallery_dataset.get_training_data(samples_per_person, samples_test=5, persons_limit=persons) # this gives persons * 5 gallery test samples

    estimators = { # this is a dict estimator like a data type (e.g return of expand_evm())
        'EVM': EVM(open_set_threshold=0.0, tail=len(train_data)),
        'KNN': KNN(n_neighbors=1)
    }

    
    reload_models=False # hmmmm i dont think i can get always the sample train/test split so lets not reload
    #reload_models=True # useful when debugging
    if reload_models:
        print("Reloading pretrained models")

        for name, estim in estimators.items():
            estimators[name] = load_model(name+'.pkl')
    else: # plain train
        print("Training raw models")

        for name, estim in estimators.items():
            
            train(estim, train_data, verbose=True)
            save_model(name+'.pkl', estim)

    # logging test
    #OSTs = [0.0, 0.001, 0.005, 0.01, 0.02]
    #test_estimators = expand_evm(estimators['EVM'], OSTs)
    #test_estimators.update({'KNN': estimators['KNN']})
    test_estimators = {'KNN': estimators['KNN'], 'EVM': estimators['EVM']}

    test_logging(test_estimators, test_data, os.path.join("%dpp" % samples_per_person, 'gallery_test.csv'))
    test_logging(test_estimators, hidden_data, os.path.join("%dpp" % samples_per_person, 'hidden_test.csv'))


def performance_eval(estim, fit_data, predict_data, n_samples:int, verbose=True):
    fit_x, fit_y = fit_data
    vprint("doing a fit", verbose)
    fit_t = measure_call(lambda: estim.fit(fit_x, fit_y), verbose=True, n_samples=5)#n_samples)
    vprint(fit_t, verbose)

    
    pred_x, _ = predict_data    
    vprint("doing a predict", verbose)
    pred_t = measure_call(lambda: estim.predict(pred_x), n_samples=n_samples)
    vprint(pred_t, verbose)

    return {'fit': fit_t, 'pred': pred_t}

def performance_multieval(estims:dict, samples_per_person:int, persons:int, n_samples:int, verbose=False):
    #hidden_dataset = vggDataset('/media/Datasets/vgg_face2', 'test')#, verbose=True)
    gallery_dataset = vggDataset('/media/Datasets/vgg_face2', 'train_aligned')

    fit_data, _ = gallery_dataset.get_training_data(samples_per_person, 0, persons)
    predict_data, _ = gallery_dataset.get_training_data(1, 0, 1)
    
    performance = {}
    for k in sorted(models.keys()):
        print("\nEvaluating %s with %d persons and %d samples-per-person (%d total samples)" % (k, persons, samples_per_person, persons*samples_per_person))
        performance.update({k: performance_eval(models[k], fit_data, predict_data, n_samples=100, verbose=verbose)})

    print_dict(performance)
    
    
if __name__ == '__main__':
    mode = 'performance'
    mode = 'experiment'

    if mode == 'performance':
        #models = {("EVM_Red_%4d" % int(r)): EVM(redundancy_rate=r/1000.) for r in range(0, 1000, 25)} # just redundancy
        models = {("EVM_BiasD_%4d" % int(r)): EVM(redundancy_rate=r/1000., biased_distance=r/1000.) for r in range(400, 1000, 50)} # biased distance (needs to change redundancy as well)

        models.update({"KNN": KNN(n_neighbors=1)})
        #print(models)

        persons = 15
        per_person_samples = 20
        performance_multieval(models, per_person_samples, persons, n_samples=100, verbose=True)

    if mode == 'experiment':
        persons = 8000
        per_person_samples = [10, 20, 50]
        
        for pp in per_person_samples:
            experiment(pp, persons)
    

