__author__ = 'Anna Kukleva'
__date__ = 'June 2019'

from hvq.corpus import Corpus
from hvq.utils.arg_pars import opt
from hvq.utils.logging_setup import logger
from hvq.utils.util_functions import timing, update_opt_str, join_return_stat, parse_return_stat

@timing
def temp_embed():
    corpus = Corpus(subaction=opt.subaction, frame_sampling=30) # loads all videos, features, and gt
    logger.debug('Corpus with poses created')
    
    # The loop below is used to train and evaluate the model for every epochs by setting opt.vqt_epochs = 1 
    for iter in range(opt.epochs):
        # Reset embedding dimension when using classifier
        opt.embed_dim = opt.f_maps

        ### Train the model ###
        # Train the MLP model as in CTE      
        if opt.model_name in ['mlp']:
            corpus.regression_training()
        # Train the HVQ model
        elif opt.model_name == 'vq':
            corpus.train_vq()
        if opt.model_name == 'nothing':
            corpus.without_temp_emed()

        ### Clusters ###
        if opt.clustering_method == 'kmean':
            corpus.clustering()
        elif opt.clustering_method == 'vq':
            corpus.vq_clustering()
        elif opt.clustering_method == 'none':
            logger.debug("No clustering")
        else:
            raise Exception('Clustering method not found')
        
        corpus.update_likelihood()
        if opt.use_cls:
            corpus.train_classifier()
        # corpus.bg_update()   

        ### Decoding ###
        if opt.decoder == 'viterbi':
            corpus.viterbi_decoding()
        elif opt.decoder == 'fifa':
            corpus.fifa_decoding()
        elif opt.decoder == "cte":
            corpus.cte_viterbi_decoding() 
        elif opt.decoder == "dummy":
            corpus.dummy_decoding()       
        else:
            logger.debug("No decoder")
            corpus.no_decoder()

        corpus.accuracy_corpus(f'VQ epochs: {iter} ')
    
    # opt.bg = True
    # corpus.update_decoding()
    corpus.accuracy_corpus('final')
    # opt.bg = False

    return corpus.return_stat


@timing
def all_actions(actions):
    return_stat_all = None
    lr_init = opt.lr
    for action in actions:
        opt.subaction = action
        if not opt.resume:
            opt.lr = lr_init
        update_opt_str()
        return_stat_single = temp_embed()
        return_stat_all = join_return_stat(return_stat_all, return_stat_single)
    logger.debug(return_stat_all)
    parse_return_stat(return_stat_all)


def resume_segmentation(iterations=10):
    logger.debug('Resume segmentation')
    corpus = Corpus(subaction=opt.action)

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.resume_segmentation()
        corpus.accuracy_corpus()
    corpus.accuracy_corpus()
