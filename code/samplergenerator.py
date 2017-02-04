







def get_sampler():
    return Sampler(
        #USE THIS:
        #graphtransformer=mol.GraphTransformerCircles(),
        #decomposer = decompose.MinorDecomposer(),

        # OR THIS:
        feasibility_checker=feasibility.cycle_feasibility_checker(6),

        #  some default stuff
        #grammar=LocalSubstitutableGraphGrammar(radius_list=[0,1], thickness_list=[1,2], min_cip_count=2,min_interface_count=2),
        # vectorizer=eden.graph.Vectorizer(complexity=3,n_jobs=2),
        # estimator=estimate.OneClassEstimator(nu=.5, cv=2, n_jobs=-1),
        # feasibility_checker=feasibility.FeasibilityChecker(),
        #graphtransformer=transform.GraphTransformer(),
        #decomposer=decompose.Decomposer(node_entity_check=lambda x, y:True, nbit=20),
        random_state=None,
        n_steps=30,
        n_samples=2,
        core_choice_byfrequency=False,
        core_choice_byscore=False,
        core_choice_bytrial=True,
        core_choice_bytrial_multiplier=1.3,
        size_diff_core_filter=4,
        burnin=10,
        include_seed=False,
        proposal_probability=False,
        improving_threshold_fraction=.5,
        improving_linear_start_fraction=0.0,
        accept_static_penalty=0.0,
        n_jobs=4,
        select_cip_max_tries=100,
        keep_duplicates=False,
        monitor=False)




def get_dict(id):
    return d[id]


def run_exp(id):
    pass



import sys
import showroc as sr
if __name__ == '__main__':
    lis = map(int, sys.argv[1:])
    if len(lis)==1:
        start=0
        end=lis[0]
    else:
        start, end = lis

    for id in range(start,end):
        d=get_dict(id)
        data=sr.make_data('1834',repeats=3,trainclass=1,train_size=30)
        stuff = sr.generative_training(data,niter=2)
        estis,newgraphs = stuff
        detailed_roc_oracle, best_graphs, quick_roc_gat, quick_roc_internal_gat= sr.evaluate_all(data, estis, newgraphs, draw_best=5)
        sr.simple_draw_graph_quality(quick_roc_gat, title='estimator quality',file='sam_%d_1' % id)
        sr.simple_draw_graph_quality(quick_roc_internal_gat, title='new graph quality',file= 'sam_%d_2' % id)

