
# SETUP ...
import logging
from eden.util import configure_logging
#configure_logging(logging.getLogger(), verbosity=2)
from IPython.core.display import HTML
HTML('<style>.container { width:95% !important; }</style><style>.output_png {display: table-cell;text-align: center;vertical-align: middle;}</style>')
from toolz import curry, compose, map, concat, pipe, first, second, take


# make a data source
assay_id = '651610' #apr93 23k mols
assay_id = '624466' #apr88
assay_id = '588350' #apr86
assay_id = '449764' #apr85
assay_id = '492952' #apr85
assay_id = '463112' #apr82
assay_id = '463213' #apr70
assay_id = '119'    #apr60 30k mols
assay_id = '1224857'#apr10
assay_id = '2326'   #apr03 200k mols

assay_id = '1834'   #apr90 500 mols
from eden_chem.io.pubchem import download
download_active = curry(download)(active=True)
download_inactive = curry(download)(active=False)


# make a vectrorizer
num_mols=4000
import multiprocessing as mp
num_cpus = mp.cpu_count()
block_size = num_mols / num_cpus

from eden.graph import Vectorizer
def vectorize(thing,**kwargs):
    v=Vectorizer(**kwargs)
    return v.transform(thing)
cvectorize = curry(vectorize)(complexity=3, nbits=20, n_jobs=num_cpus, block_size=block_size)


"""
# make a data source 
def selection_iterator( ids, iterable=None):
    '''itertor from eden.. but better curryability'''
    ids = sorted(ids)
    counter = 0
    for id, item in enumerate(iterable):
        if id == ids[counter]:
            yield item
            counter += 1
            if counter == len(ids):
                break
graphs=pipe(download_active(assay_id),babel_load)
selectgraphs = curry(selection_iterator)(iterable=list(graphs))
"""

import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
#from eden_chem.obabel import load as babel_load  BABEL IS BULL
from GArDen.convert.molecule import sdf_to_nx as babel_load
def make_data(assay_id):
    active_X = pipe(assay_id, download_active, babel_load, cvectorize)
    inactive_X = pipe(assay_id, download_inactive, babel_load, cvectorize)
    X = vstack((active_X, inactive_X))
    y = np.array([1]*active_X.shape[0] + [-1]*inactive_X.shape[0])
    graphs=list(pipe(assay_id, download_active,babel_load))
    return X, y, graphs 

def test(estimator, X):
        y_pred = estimator.predict(X)
        #y_score = estimator.decision_function(X)
        y_score = estimator.predict_proba(X)[:,0]
        return y_pred, y_score

# do something with the data... 
# also add a loop 

from eden.util import selection_iterator
from graphlearn.trial_samplers import GAT
def train_and_test( data, train_size, trainclass=1,niter=20): # trainclass needs to be 1 ... 

    # select data, train, report 
    X, y, graphs = data
    z = np.where(y==trainclass)[0]
    ids = np.random.permutation(z)[:train_size]
    
    usegraphs = list(selection_iterator( graphs, ids.tolist()))
    
    
    #print '$$$$$$IDS:',ids.tolist(), len(usegraphs)
    

    # use rest for testing 
    every = set(range(X.shape[0]))
    want = np.array( list( every-set(ids)))
    X_test = X[want]
    Y_test = y[want]

    # train 
    estis, congraph,seedzz = GAT.sample(GAT.get_sampler(),
                                        n_iterations=niter,
                                        seedgraphs=usegraphs)


    result=[]
    # report
    for i,e in enumerate(estis[1:]):
        y_pred, y_score = test(e, X_test)
        result.append( [Y_test,y_pred,y_score] )
    #y_test should be available anyway... 

    return  result #(should be a list of lists)


def collect_data(repeats=3, train_size=100,niter=20):
    result = []
    for e in range(repeats):
        data=make_data(assay_id)
    # for i in range(repeats) # add loop later
        result.append( train_and_test(data,train_size,niter=niter) )


    # transpose [123][123][123] => [111][222][333]
    return map(list, zip(*result))





# DISPLAY THE THINGS 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from eden_display import plot_confusion_matrices
from eden_display import plot_aucs
def predicitve_performance_report(data,filename=None):
    y_true, y_pred, y_score = data
    line_size = 135
    print '_'*line_size
    print
    print 'Accuracy: %.2f' % accuracy_score(y_true, y_pred)
    print ' AUC ROC: %.2f' % roc_auc_score(y_true, y_score)
    print '  AUC AP: %.2f' % average_precision_score(y_true, y_score)
    print '_'*line_size
    print
    print 'Classification Report:'
    print classification_report(y_true, y_pred)
    print '_'*line_size
    print
    plot_confusion_matrices(y_true, y_pred, size=int(len(set(y_true))*2.5),filename='confus_%d_.png' % filename)
    print '_'*line_size
    print
    plot_aucs(y_true, y_score, size=10,filename='auc_%d_.png' % filename)



def prepdata(data):
    for bunch in data:
        sortedbunch = map(list, zip(*bunch))
        res = [ np.hstack(tuple(thing)) for thing in sortedbunch]
        yield res

for i, item in enumerate (prepdata(collect_data(repeats=3,train_size=100,niter=20))):
    predicitve_performance_report(item,i)
