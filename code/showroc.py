# SETUP ...
from toolz import curry, compose, map, concat, pipe, first, second, take

from eden_chem.io.pubchem import download

download_active = curry(download)(active=True)
download_inactive = curry(download)(active=False)

# make a vectrorizer
num_mols = 4000
import multiprocessing as mp

num_cpus = mp.cpu_count()
block_size = num_mols / num_cpus

from eden.graph import Vectorizer


def vectorize(thing, **kwargs):
    v = Vectorizer(**kwargs)
    return v.transform(thing)


cvectorize = curry(vectorize)(complexity=3, nbits=20, n_jobs=num_cpus, block_size=block_size)

import numpy as np
from scipy.sparse import vstack
from GArDen.convert.molecule import sdf_to_nx as babel_load  # !!!


def make_data(assay_id):
    active_X = pipe(assay_id, download_active, babel_load, cvectorize)
    inactive_X = pipe(assay_id, download_inactive, babel_load, cvectorize)
    X = vstack((active_X, inactive_X))
    y = np.array([1] * active_X.shape[0] + [-1] * inactive_X.shape[0])
    graphs = list(pipe(assay_id, download_active, babel_load)) + list(pipe(assay_id, download_inactive, babel_load))
    return X, y, graphs


def test(estimator, X):
    y_pred = estimator.predict(X)
    # y_score = estimator.decision_function(X)
    y_score = estimator.predict_proba(X)[:, 0]
    return [y_pred, y_score]


from eden.util import selection_iterator
from graphlearn.trial_samplers import GAT


def train_and_test(data, train_size, trainclass=1, niter=20):
    # select data ->  train -> test
    X, y, graphs = data
    possible_train_ids = np.where(y == trainclass)[0]
    train_ids = np.random.permutation(possible_train_ids)[:train_size]
    train_graphs = list(selection_iterator(graphs, train_ids.tolist()))

    # rename things
    # create test data..
    all_ids = set(range(X.shape[0]))
    test_ids = np.array(list(all_ids - set(train_ids)))
    X_test = X[test_ids]
    Y_test = y[test_ids]

    # train ...
    estimators, constructed_graphs = GAT.generative_adersarial_training(
                                         GAT.get_sampler(),
                                         n_iterations=niter,
                                         seedgraphs=train_graphs)
    # test
    return map( lambda x:[Y_test]+test(x,X_test),estimators[1:] )


def transpose_and_hstack(data):
    # eg [111] is actually a <t,s,p> <t,s,p> etc   => transform these to this:
    # stack (t1,t2,t3..) , stack (s1,s2,s3...) , stack(p1,p2 .. )
    # (t s p) is the  true y, the score of y and the predicted class ...
    # the actual order is btw <t p s >
    transposed_data = map(list, zip(*data))
    return [np.hstack(tuple(thing)) for thing in transposed_data]



def collect_data(assay_id=None, repeats=3, train_size=100, niter=20):
    result = [train_and_test(make_data(assay_id),train_size,niter=niter) for i in range(repeats)]
    # transpose [123][123][123] => [111][222][333]
    # ( where the numbers indicate the level of adversaries :)
    result = map(list, zip(*result))
    return map(transpose_and_hstack, result)



# DISPLAY THE THINGS
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from eden_display import plot_confusion_matrices
from eden_display import plot_aucs


def predicitve_performance_report(data, filename=None):
    y_true, y_pred, y_score = data
    line_size = 135
    print '_' * line_size
    print
    print 'Accuracy: %.2f' % accuracy_score(y_true, y_pred)
    print ' AUC ROC: %.2f' % roc_auc_score(y_true, y_score)
    print '  AUC AP: %.2f' % average_precision_score(y_true, y_score)
    print '_' * line_size
    print
    print 'Classification Report:'
    print classification_report(y_true, y_pred)
    print '_' * line_size
    print
    plot_confusion_matrices(y_true, y_pred, size=int(len(set(y_true)) * 2.5), filename='confus_%d_.png' % filename)
    print '_' * line_size
    print
    plot_aucs(y_true, y_score, size=10, filename='auc_%d_.png' % filename)


# make a data source
assay_id = '651610'  # apr93 23k mols
assay_id = '624466'  # apr88
assay_id = '588350'  # apr86
assay_id = '449764'  # apr85
assay_id = '492952'  # apr85
assay_id = '463112'  # apr82
assay_id = '463213'  # apr70
assay_id = '119'  # apr60 30k mols
assay_id = '1224857'  # apr10
assay_id = '2326'  # apr03 200k mols
assay_id = '1834'  # apr90 500 mols

for i, item in enumerate(collect_data(assay_id=assay_id, repeats=2, train_size=50, niter=2)):
    predicitve_performance_report(item, i)
