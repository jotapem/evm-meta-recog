# hmmm csv from predictions become a csv with simple metrics and a x-axe

import os
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from csv_eval import load_content

def column_from_csv(content, column):
    return list(map(lambda x: float(x[column]), content))

def lazy_thresh_plot(experiment_folder, target_csv, y_keys, x_key='x', scale='linear'):
    results_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(results_path, experiment_folder)
    graph_name = ' '.join([experiment_folder] + target_csv.split('.')[-2].split('_'))
    print(graph_name)

    content = load_content(os.path.join(results_path, target_csv))
    content.sort(key=lambda x: x[x_key])

    fig = plt.figure()
    ax = plt.subplot(111)
    for y_key in y_keys:
        ax.plot(list(map(lambda x: float(x), column_from_csv(content, x_key))),
                list(map(lambda x: float(x), column_from_csv(content, y_key))),
                label=y_key)

    plt.title(graph_name)

    ax.legend()
    fig.savefig(os.path.join(results_path, "%s.%s" % (graph_name.replace(' ', '_'), 'png')))
    plt.close(fig)

def plain_thresh(exp, estim, test):
    """
    Threshold as x-axis (already tagged as 'x' column)
    y-axis depends on the kind of test
    """
    
    target_csv = ("%s_%s_test.csv" % (estim, test)).lower()

    keys = {
        'gallery': ['tp', 'fp', 'fn'],
        'hidden': ['tn', 'fp']
    }[test]

    lazy_thresh_plot(exp, target_csv, keys)

def plot_curves(contents:dict, out_path:str, curve_type:str):
    """
    Contents is a dictionary where keys are the estimator name (plot label) and the values are predictions rows
    """

    fig = plt.figure()
    ax = plt.subplot(111)
    for estim_name, content in contents.items():
        x_key = 'x'
        y_key = 'y'
        
        ax.plot(list(map(lambda x: float(x), column_from_csv(content, x_key))),
                list(map(lambda x: float(x), column_from_csv(content, y_key))),
                label=estim_name)
    ax.legend()
    
    plt.ylim([0,1])

    if curve_type == 'roc':
        plt.title('ROC (Correctly Identified vs Mistakenly Identified) curve')
    elif curve_type == 'crr':
        plt.title('CRR (Correctly Rejected vs Correctly Identified) curve')
        

    fig.savefig(out_path)
    plt.close(fig)

def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    estims = ['knn', 'evm']
    
    for exp in ['10pp', '20pp', '50pp']:
        # ROC / CRR plots
        for curve in ['roc', 'crr']:
            curve_contents = {estim: load_content(os.path.join(file_dir, exp, '%s_%s.csv' % (estim, curve))) for estim in estims}
            plot_curves(curve_contents, os.path.join(file_dir, exp, '%s.png' % curve), curve)

        # Open Set / Euclidean Distance threshold plots
        for test in ['gallery', 'hidden']:
            for estim in estims:
                plain_thresh(exp, estim, test)


if __name__ == '__main__':
    main()
