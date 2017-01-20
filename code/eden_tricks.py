# this is called eden tricks, because it is taken straight out of an eden notebook :)
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
import  sklearn.model_selection as modsel

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDClassifier



# cross val from ze library but betta.
import sklearn
def cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):

    X, y, groups = sklearn.utils.indexable(X, y, groups)

    cv = sklearn.model_selection._split.check_cv(cv, y, classifier=sklearn.base.is_classifier(estimator))
    cv_iter = list(cv.split(X, y, groups))
    scorer = sklearn.metrics.scorer.check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = sklearn.externals.joblib.Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(sklearn.externals.joblib.delayed(sklearn.model_selection._validation._fit_and_score)
                      (sklearn.base.clone(estimator), X, y, scorer,
                                              train, test, verbose, None,
                                              fit_params,return_train_score=True)
                      for train, test in cv_iter)
    return np.array(scores) #[:, 0]





def predictive_performance(estimator, data):
    X, y = data
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    scoring = make_scorer(accuracy_score)


    #print  cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)


    return scores[:,0],scores[:,1]





def task_difficulty(X,y):

    sgd = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=4)
    scores_train, scores_test = predictive_performance(sgd,(X,y))

    #print scores_train, scores_test
    return 'crossval train: %.4f +- %.4f   crossval: %.4f +- %.4f' % (np.mean(scores_train),np.std(scores_train),
                                             np.mean(scores_test),np.std(scores_test))
