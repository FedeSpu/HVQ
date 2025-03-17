# import sys
# sys.path.append('/home/fspurio/Projects/Papers/HVQ/')

from hvq.utils.arg_pars import opt
from data_utils.BF_utils.update_argpars import update
from hvq.ute_pipeline import temp_embed, all_actions
from os.path import join


if __name__ == '__main__':
    opt.seed = 42
    
    #### Set root and paths ####
    opt.dataset_root = './data/breakfast'
    opt.original_feat_path = join(opt.dataset_root, 'features') 
    opt.vq_model_path = join(opt.dataset_root, 'embed', f'embed_{opt.subaction}')
    opt.ext = 'txt'    # Set feature extension

    #### Set activity ####
    opt.subaction = 'all'  # ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'salat']
    
    #### Resume training or load a model ####
    opt.resume = False          # Resume training    
    opt.load_model = False      # Load an already trained model (stored in the models directory in dataset_root)
    opt.loaded_model_name = '%s.pth.tar'
    opt.load_embed_feat = False  # Load the embedding features (stored in the embed directory in dataset_root)
    opt.embed_path_suff = "my_embed" 
    
    opt.model_name = 'vq'           # Accepted arguments: 'mlp', 'vq' or 'nothing' 
                                    # for no embedding (just raw features)    
    opt.clustering_method = 'vq'    # Accepted arguments: 'vq', 'kmean', 'none'
    
    opt.feature_dim = 64            # Input feature dimension
    opt.embed_dim = 32              # Embedding dimension (can change during training
                                    # if opt.use_cls and opt.update_w_cls_emb are set)
    opt.f_maps = 32                 # Hidden dimension of the model
    
    opt.force_update_z = True       # Force the storing of predicted labels     
    opt.use_scheduler = False       # # Use scheduler during training for lr

    #### MS-TCN training parameters ####
    opt.num_stages = 2
    opt.dropout = 0
    
    #### VQ training parameters ####    
    opt.model_type = "double"       # Accepted arguments: single, double, triple 
    opt.vq_dim = 32                 # Internal dimension of prototypes
    opt.vqt_input_dim = 64          # Input dimension for the VQ model    
    opt.vq_class_multiplier = 2     # alpha parameter: len(Z) = alpha * len(Q)
    # Epochs for training the VQ model:
    # - if opt.vqt_epochs=20 and opt.epochs=1, evaluation only on the last epoch
    # - if opt.vqt_epochs=1 and opt.epochs=20, evaluation on all epochs
    opt.vqt_epochs = 20
    opt.epochs = 1
    
    opt.vqt_lr = 1e-3       # Learning rate for the VQ model
    opt.gumbel = False      # Use gumbel noise for selecting prototypes
    # Use EMA or VQ loss. They should be exclusive   
    opt.vq_loss = False
    opt.use_ema = True
    opt.vq_decay = 0.8      # Weight decay for EMA update

    # If there are more than 3 unused prototypes, 
    # they are removed and init. again with a random embedding
    opt.ema_dead_code = 3
    
    opt.vq_norm = True      # Normalize input features before HVQ    
    opt.f_norm = False      # Normalize the features while loading them (not suggested)
    
    opt.vq_kmeans = True    # Init. of prototypes with KMeans

    #### VQ loss ####
    opt.vqt_commit_weight = 1
    opt.vqt_rec_weight = 0.002 

    # Distance Loss for prototypes in Q
    opt.dist_loss = False
    opt.dist_margin = 0.5
    opt.dist_weight = 0.1

    #### FIFA parameters ####
    opt.decoder = "fifa"    
    opt.fifa_weight = 0.001
    opt.fifa_epochs = 100
    opt.fifa_sharpness = 0.1 
    opt.fifa_lr = 0.6  
    opt.fifa_use_adam = False 

    #### Use an MLP after VQ assignment ####
    # (it makes the predictions more stable during training)
    opt.use_cls = False
    opt.update_w_cls_emb = True    
    opt.cls_epochs = 15
    opt.use_gt_cls = False      # Use gt labels for training the classifier (only for debugging)
    opt.lr = 1e-5               # Learning rate for the classifier

    #### Visualization options ####
    opt.vis = False
    opt.vis_mode = 'segm'  
    
    # update log name and absolute paths
    update()

    # run temporal embedding
    if opt.subaction == 'all':
        actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat',
                   'pancake']
        all_actions(actions)
    elif isinstance(opt.subaction, list):
        all_actions(opt.subaction)
    else:
        temp_embed()
