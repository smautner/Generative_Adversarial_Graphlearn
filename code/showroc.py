from toolz import curry, compose, concat, pipe, first, second, take
import matplotlib
matplotlib.use('Agg')
from eden_chem.io.pubchem import download
from eden.graph import Vectorizer
import numpy as np
from scipy.sparse import vstack
from eden_chem.io.rdkitutils import sdf_to_nx as babel_load  # !!!
from eden_chem.display.rdkitutils import nx_to_image
from eden.util import selection_iterator
from graphlearn.trial_samplers import GAT
# DISPLAY IMPORTS
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from eden_display import plot_confusion_matrices
from eden_display import plot_aucs
from sklearn.linear_model import SGDClassifier
import eden_tricks




import matplotlib.pyplot as plt
download_active = curry(download)(active=True,stepsize=500) # default stepsize = 50 (way to few)
download_inactive = curry(download)(active=False,stepsize=500)


def vectorize(thing):
    v = Vectorizer()
    return v.transform(thing)

def transpose(things):
    return map(list,zip(*things))

def test(estimator, X):
    y_pred = estimator.predict(X)
    y_score = estimator.predict_proba(X)[:, 0]
    return [y_pred, y_score]

def error(esti,X,y):
    pred = esti.predict(X) == y
    return float(sum(pred))/ len(y)





'''
There is going to be a main that is

1. make data [(test),test_trained_esti,train_graphs] for each repeat

2. conducting training   train_graphs -> newestisestis, newgraphs

3. evaluate things roc, graph_quality, select graphs
4. draw   roc, quality of graphs, some newgraphs
'''

def get_data(assay_id):
    active_X = pipe(assay_id, download_active, babel_load, vectorize)
    inactive_X = pipe(assay_id, download_inactive, babel_load, vectorize)

    X = vstack((active_X, inactive_X))
    y = np.array([1] * active_X.shape[0] + [-1] * inactive_X.shape[0])
    graphs = list(pipe(assay_id, download_active, babel_load)) + list(pipe(assay_id, download_inactive, babel_load))
    stats={'active':active_X.shape[0], 'inactive':inactive_X.shape[0]}

    return X, y, graphs, stats


def make_data(assay_id,repeats=3,trainclass=1,train_size=50, not_train_class=-1, test_size_per_class=100):
    #   [(test), test_trained_esti, train_graphs] for each repeat

    X,y,graphs,stats= get_data(assay_id)
    print 'indicator of tak-ease:'
    print eden_tricks.task_difficulty(X,y)
    print stats

    esti = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4, loss='log')
    esti.fit(X,y)

    def get_run():
        # get train items
        possible_train_ids = np.where(y == trainclass)[0]
        train_ids = np.random.permutation(possible_train_ids)[:train_size]
        train_graphs = list(selection_iterator(graphs, train_ids.tolist()))

        # get test items
        possible_test_ids_1 = np.array(list( set(possible_train_ids) - set(train_ids)))
        possible_test_ids_0 = np.where(y == not_train_class)[0]
        test_ids_1 = np.random.permutation(possible_test_ids_1)[:test_size_per_class]
        test_ids_0 = np.random.permutation(possible_test_ids_0)[:test_size_per_class]
        test_ids= np.hstack((test_ids_1,test_ids_0))
        X_test = X[test_ids]
        Y_test = y[test_ids]

        #esti= SGDClassifier(loss='log')
        #esti.fit(X_test,Y_test)
        return {'X_test':X_test,'y_test':Y_test,'oracle':esti,'graphs_train':train_graphs}

    return [get_run() for i in range(repeats)]


############################################################################

def generative_training(data,niter):
    # data -> [estis]*niter, [gengraphs]*niter
    train= lambda x: GAT.generative_adersarial_training(
        GAT.get_sampler(), n_iterations=niter, seedgraphs=x, partial_estimator=False)
    stuff = [ train(x['graphs_train']) for x in data ]
    return transpose(stuff)

#########################################################################


def roc_data(estis,data):
    # test each generated estimator with the according estimator
    dat = [ map(lambda x: [dat['y_test']]+test(x,dat['X_test']) ,esti ) for esti,dat in zip(estis,data)]
    # transposing orders the result by level or depth or whatever.
    # the map above is generating a 2d matrix -> we have $repeats many
    # we can just hstack the 2d arrays..
    return [np.hstack(tuple(allrepeats)) for allrepeats in transpose(dat)]

def drawroc_data(rocdata):
    for e in rocdata:
        predicitve_performance_report(e)

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
    if filename:
        conf_filename = 'confus_%d_.png' % filename
        auc_filename = 'auc_%d_.png' % filename
    else:
        conf_filename, auc_filename = None, None
    plot_confusion_matrices(y_true, y_pred, size=int(len(set(y_true)) * 2.5), filename=conf_filename)
    print '_' * line_size
    print
    plot_aucs(y_true, y_score, size=10, filename=auc_filename)
##

def select_graphs(graphs,estis, print_best=5 ):
    if print_best > 0:
        # calculate the scores of the graphs  with the right estimators... GATdepth*repeats
        scores = [[e.predict(vectorize(g)) for g,e in zip(gs,es)]  for gs,es in zip(graphs,estis)]
        # take the graphs with the best scores ..
        graphs = [[ list(selection_iterator(gr,np.argpartition(sco,-print_best)[-print_best:]))
                    for gr,sco in zip(grs,scores)]   for grs,scores in zip(graphs,scores)]
        # collapse graphs that are on the same GAT-level
        return map(lambda x: reduce(lambda y,z: z+y,x),transpose(graphs))




from graphlearn.utils import molecule
def draw_select_graphs(graphs):
    for i, graphlist in enumerate(graphs):
        print 'best graphs (according to GAT) for repeat #%d' % i
        molecule.draw(graphlist)
        #pic = nx_to_image(graphlist)
        #plt.figure()
        #plt.imshow(np.asarray(pic))

##
def test_vs_gatesties(estis,data):

    scores = [[error(level_esti,repeats_data['X_test'],repeats_data['y_test'])# level_est and repeats_data
               for level_esti in repeats_es]
                    for repeats_data, repeats_es in zip(data, estis)]

    # scores are now [l1,l2,l3..][l1,l2,l3..][l1,l2,l3..]
    # scores = map(lambda z: reduce(lambda x,y: np.concatenate((x,y)),z) , transpose(scores))
    #print scores
    scores = transpose(scores)
    # transpose to get: [l1,l1,l1..][l2,l2,l2..]  .. and we just flatten the [l_x,l_x..]
    return transpose([ [np.mean(e),np.std(e)] for e in scores])

##
def internalgat(estis, newgraphs):
    scores = [[level_esti.predict_proba(vectorize(level))[:,0] for level,level_esti in zip(repeats,repeats_es[1:])] for repeats, repeats_es in zip(newgraphs, estis)]
    # scores are now [l1,l2,l3..][l1,l2,l3..][l1,l2,l3..]
    scores = map(lambda z: reduce(lambda x,y: np.concatenate((x,y)),z) , transpose(scores))
    # transpose to get: [l1,l1,l1..][l2,l2,l2..]  .. and we just flatten the [l_x,l_x..]
    return transpose([ [np.mean(e),np.std(e)] for e in scores])

##
def graphlol(data, newgraphs):
    estis= [d['oracle'] for d in data]
    scores = [[es.predict_proba(vectorize(level))[:,0] for level in repeats] for repeats, es in zip(newgraphs, estis)]
    # order by level and concatenate over all repeats
    scores = map(lambda z: reduce(lambda x,y: np.concatenate((x,y)),z) , transpose(scores))
    # get means snd stds
    return transpose([ [np.mean(e),np.std(e)] for e in scores])




def draw_graph_quality(data):
    means,stds= data
    plt.figure(figsize=(14, 5))
    fig, ax = plt.subplots()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(14)
    #plt.ylim(0, 80)
    #plt.xlim(0, 400)
    plt.axhline(y=38, color='black', linewidth=3)

    def fillthing(y, std, label='some label', col='b'):
        y = np.array(y)
        std = np.array(std)
        ax.fill_between('asd', y + std, y - std, facecolor=col, alpha=0.3, linewidth=0)
        # ax.plot(labels,y,label=label,color='gray')
        ax.plot('asd', y, color='gray')

    fillthing(means, stds, col='#6A9AE2')

    ax.plot('asdasd', means, label='new CIP', color='b', linewidth=2.0)
    # add some text for labels, title and axes ticks
    labelfs = 16
    ax.set_ylabel('something', fontsize=labelfs)
    ax.set_xlabel('something2', fontsize=labelfs)
    ax.legend(loc='lower right')
    plt.show()

def simple_draw_graph_quality(data,title='title',file=None):
    means,std = data
    plt.figure()
    plt.errorbar(range(len(means)), means,  yerr=std)
    plt.title(title)
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()

##


def evaluate_all(data,estis,newgraphs,draw_best=5):
    rocdata = roc_data(estis,data)            # what does this need to look like?
    graphs =  select_graphs(newgraphs, estis, print_best=draw_best )      # select some graphs that need drawing later
    graph_quality = graphlol(data,newgraphs)
    graph_quality_internal = internalgat(estis,newgraphs)
    res5 = test_vs_gatesties(estis,data)
    return rocdata,graphs,graph_quality, graph_quality_internal, res5




######################################################################



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




if __name__ == '__main__':
    data=make_data(assay_id,repeats=3,trainclass=1,train_size=30)
    stuff = generative_training(data,niter=2)
    estis,newgraphs = stuff
    detailed_roc_oracle, best_graphs, quick_roc_gat, quick_roc_internal_gat= evaluate_all(data, estis, newgraphs, draw_best=5)
    simple_draw_graph_quality(quick_roc_gat, title='estimator quality',file='1')
    simple_draw_graph_quality(quick_roc_internal_gat, title='new graph quality',file='2')
    #draw_select_graphs(best_graphs)
    #drawroc_data(detailed_roc_oracle)
