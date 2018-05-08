# hmmm csv from predictions become a csv with simple metrics and a x-axe

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from csv_eval import load_content

def column_from_csv(content, column):
    return list(map(lambda x: float(x[column]), content))

def grosso(experiment_folder, target_csv, y_keys):
    results_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(results_path, experiment_folder)
    graph_name = ' '.join([experiment_folder] + target_csv.split('.')[-2].split('_'))
    print(graph_name)

    content = load_content(os.path.join(results_path, target_csv))
    content.sort(key=lambda x: x['x'])

    #X = list(map(lambda x: float(x['x']), content))
    #Y = list(map(lambda x: float(x[y_key]), content))
    #print(X) ; print(Y)

    fig = plt.figure()
    ax = plt.subplot(111)
    for y_key in y_keys:
        ax.plot(column_from_csv(content, 'x'),
                column_from_csv(content, y_key),
                label=y_key)
    plt.title(graph_name)
    ax.legend()

    fig.savefig(os.path.join(results_path, "%s.%s" % (graph_name, 'png')))


def main():
    for exp in ['10pp', '20pp', '50pp']:
        for estim in ['knn', 'evm']:
            for test in ['gallery', 'hidden']:
                target_csv = "%s_%s_test.csv" % (estim, test)

                keys = {'gallery': ['tp', 'fp', 'fn'],
                        'hidden': ['tn', 'fp']
                }[test]

                grosso(exp, target_csv, keys)


if __name__ == '__main__':
    main()
