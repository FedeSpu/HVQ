import numpy as np
import random
import os
from os.path import join
import os.path as ops
import torch
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

from hvq.video import Video
from hvq.models import mlp
from hvq.models import cls
from hvq.utils.arg_pars import opt
from hvq.utils.logging_setup import logger
from hvq.eval_utils.accuracy_class import Accuracy
from hvq.utils.mapping import GroundTruth
from hvq.utils.util_functions import join_data, timing, dir_check
from hvq.utils.visualization import Visual, plot_segm
from hvq.eval_utils.f1_score import F1Score
from hvq.models.dataset_loader import load_reltime, load_pseudo_gt, load_single_video
from hvq.models.training_embed import load_model, training, training_cls
from hvq.viterbi_utils.grammar import SingleTranscriptGrammar
from hvq.viterbi_utils.length_model import PoissonModel
from hvq.viterbi_utils.viterbi_w_lenth import Viterbi
from hvq.viterbi_utils.fifa import fifa

from hvq.models.hvq_model import *
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(opt.seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
torch.use_deterministic_algorithms(True, warn_only=True)
# torch.use_deterministic_algorithms(True) # , warn_only=True)

class Buffer(object):
    def __init__(self, buffer_size, n_classes):
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, features, transcript, framelabels):
        if len(self.features) < self.buffer_size:
            # sequence data 
            self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            # statistics for prior and mean lengths
            self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            # statistics for prior and mean lengths
            self.instance_counts[self.next_position] = np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size

class Corpus(object):
    def __init__(self, subaction='coffee', K=None, buffer_size=2000, frame_sampling=30):
        """
        Args:
            subaction: current name of high level activity
            K: number of actions
            buffer_size: size of buffer for prior and mean lengths (for Viterbi)
            frame_sampling: number of frames to skip (for Viterbi)
        """
        self.gt_map = GroundTruth(frequency=opt.frame_frequency)
        self.gt_map.load_mapping()
        self._K = self.gt_map.define_K(subaction=subaction) if K is None else K
        logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.iter = 0
        self.return_stat = {}
        self._frame_sampling = frame_sampling

        self._acc_old = 0
        self._videos = []
        self._subaction = subaction
        # init with ones for consistency with first measurement of MoF
        self._subact_counter = np.ones(self._K)
        self._gaussians = {}
        self._inv_count_stat = np.zeros(self._K)
        self._embedding = None
        self._gt2label = None
        self._label2gt = {}

        self._with_bg = opt.bg
        self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features = None
        self._embedded_feat = None
        self._init_videos()

        # to save segmentation of the videos
        dir_check(os.path.join(opt.output_dir, 'segmentation'))
        dir_check(os.path.join(opt.output_dir, 'likelihood'))
        self.vis = None  # visualization tool

        # Viterbi decoding with length model and grammar
        self.decoder = Viterbi(None, None, self._frame_sampling, max_hypotheses = 100000) # np.inf)
        self.vq_model_trained = None
        self.pretrained_model = None

        self.buffer = Buffer(buffer_size, self._K)     
        self.mean_lengths = np.ones((self._K), dtype=np.float32) * self._frame_sampling * 2
        self.prior = np.ones((self._K), dtype=np.float32) / self._K

    def _init_videos(self):
        logger.debug('.')
        gt_stat = Counter()
        for root, dirs, files in os.walk(opt.data):
            # Skip embed features in features path
            if 'embed' in root.split("/features")[1]:
                continue
            if not files:
                continue
            
            for filename in files:
                # Pick only videos relatives to the current activity
                if self._subaction in filename:
                    match = re.match(r'(.*)\..*', filename)
                    gt_name = match.group(1)
                    # Use either extracted embeddings from pretrained on gt features
                    if opt.load_embed_feat:
                        path = os.path.join(opt.data, f'embed_{opt.embed_path_suff}' , opt.subaction, gt_name)
                    else:
                        path = os.path.join(root, filename)
                    start = 0 if self._features is None else self._features.shape[0]
                    try:
                        video = Video(path, K=self._K,
                                      gt=self.gt_map.gt[gt_name],
                                      name=gt_name,
                                      start=start,
                                      with_bg=self._with_bg)
                    except AssertionError:
                        logger.debug('Assertion Error: %s' % gt_name)
                        continue
                    self._features = join_data(self._features, video.features(), np.vstack)

                    video.reset()  # to not store second time loaded features
                    self._videos.append(video)
                    # Accumulate statistic for inverse counts vector for each video
                    gt_stat.update(self.gt_map.gt[gt_name])
                    if opt.reduced:
                        if len(self._videos) > opt.reduced:
                            break

        # Update global range within the current collection for each video
        for video in self._videos:
            video.update_indexes(len(self._features))
        logger.debug('gt statistic: %d videos ' % len(self._videos) + str(gt_stat))
        self._update_fg_mask()

    def _update_fg_mask(self):
        logger.debug('.')
        if self._with_bg:
            self._total_fg_mask = np.zeros(len(self._features), dtype=bool)
            for video in self._videos:
                self._total_fg_mask[np.nonzero(video.global_range)[0][video.fg_mask]] = True
        else:
            self._total_fg_mask = np.ones(len(self._features), dtype=bool)

    def __len__(self):
        return len(self._videos)

    def regression_training(self):
        if opt.load_embed_feat:
            logger.debug('load precomputed features')
            self._embedded_feat = self._features
            return

        logger.debug('.')

        dataloader = load_reltime(videos=self._videos,
                                  features=self._features)

        model, loss, optimizer = mlp.create_model()
        if opt.load_model:
            model.load_state_dict(load_model())
            logger.debug('load model')
            self._embedding = model
        else:
            self._embedding = training(dataloader, opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name=opt.model_name)

        self._embedding = self._embedding.cpu()

        unshuffled_dataloader = load_reltime(videos=self._videos,
                                             features=self._features,
                                             shuffle=False)

        gt_relative_time = None
        relative_time = None
        if opt.model_name == 'mlp':
            for batch_features, batch_gtreltime in unshuffled_dataloader:
                if self._embedded_feat is None:
                    self._embedded_feat = batch_features
                else:
                    self._embedded_feat = torch.cat((self._embedded_feat, batch_features), 0)

                batch_gtreltime = batch_gtreltime.numpy().reshape((-1, 1))
                gt_relative_time = join_data(gt_relative_time, batch_gtreltime, np.vstack)

            relative_time = self._embedding(self._embedded_feat.float()).detach().numpy().reshape((-1, 1))

            self._embedded_feat = self._embedding.embedded(self._embedded_feat.float()).detach().numpy()
            self._embedded_feat = np.squeeze(self._embedded_feat)

        if opt.save_embed_feat:
            self.save_embed_feat()

        mse = np.sum((gt_relative_time - relative_time)**2)
        mse = mse / len(relative_time)
        logger.debug('MLP training: MSE: %f' % mse)

    def create_cls(self):
        # return model, loss, optimizer
        return cls.create_model(self._K)

    def train_classifier(self, model=None, loss=None, optimizer=None, video=None):
        logger.debug('train framewise classifier')
        # train_classifier
        logger.debug('.')

        if video == None:
            pseudo_gt = self.real_gt if opt.use_gt_cls else self.pseudo_gt_with_bg
            dataloader = load_pseudo_gt(videos=self._videos,
                                    features=self._embedded_feat,
                                    pseudo_gt=pseudo_gt)
            num_epoch = opt.cls_epochs
        else:
            dataloader = load_single_video(videos=self._videos,
                                    features=self._embedded_feat,
                                    pseudo_gt=self.pseudo_gt_with_bg,
                                    video=video)
            num_epoch = 5

        if model==None and loss==None and optimizer==None:
            model, loss, optimizer = cls.create_model(self._K)

        self._classifier = training_cls(dataloader, num_epoch,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name=opt.model_name)
        # update video likelihood
        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid(video_idx)
            if opt.update_w_cls_emb:
                self._update_video_features(video_idx)

        if opt.update_w_cls_emb:
            self._features = None
            self._embedded_feat = None

            for video in self._videos:
                self._features = join_data(self._features, video.features(), np.vstack)                
                self._embedded_feat = join_data(self._embedded_feat, video.features(), np.vstack)
                
            opt.embed_dim = opt.embed_dim * 2

    def _load_original_features(self, path, file_name):
        '''
        Load original features from the file. This ensure that the features are not changed during training
        '''
        if opt.ext == 'npy':    
            features = np.load(os.path.join(path, file_name + '.npy'))
            if opt.subaction == 'IDU':
                features = features.T
        else:
            features = np.loadtxt(os.path.join(path, file_name + '.txt'))                
            
        features = torch.from_numpy(features).type(torch.float32).T.unsqueeze(0).to(opt.device)
        if opt.vq_norm:
            features = F.normalize(features) 

        return features
    
    def train_vq(self):     
        # The features are loaded from the original features path
        # to avoid any changes in the features during training
        features_path = os.path.join(opt.original_feat_path, self._subaction) 
        num_classes = self._K  
        
        # Initialize the HVQ model with different levels of hierarchies
        if opt.model_type == "single":
            model = SingleVQModel(num_stages=opt.num_stages, num_layers=opt.num_layers, num_f_maps=opt.f_maps, dim=opt.vqt_input_dim, num_classes=self._K, latent_dim=opt.f_maps).to(opt.device)
        elif opt.model_type == "double":
            model = DoubleVQModel(num_stages=opt.num_stages, num_layers=opt.num_layers, num_f_maps=opt.f_maps, dim=opt.vqt_input_dim, num_classes=num_classes, latent_dim=opt.f_maps, ema_dead_code=opt.ema_dead_code).to(opt.device)
        elif opt.model_type == "triple":
            model = TripleVQModel(num_stages=opt.num_stages, num_layers=opt.num_layers, num_f_maps=opt.f_maps, dim=opt.vqt_input_dim, num_classes=num_classes, latent_dim=opt.f_maps, ema_dead_code=opt.ema_dead_code).to(opt.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.vqt_lr, weight_decay=1e-4) 
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        epochs = opt.vqt_epochs
        
        # If a model is already trained, load it
        # Used especially when evaluating the model every epoch
        if self.vq_model_trained is not None:
            model.load_state_dict(self.vq_model_trained.state_dict())
            optimizer.load_state_dict(self.vq_optim.state_dict())
            epochs = 1
        elif self.pretrained_model is not None:
            model.load_state_dict(self.pretrained_model.state_dict(), strict=False)
            
        # Reconstruction loss
        loss_rec_fn = torch.nn.MSELoss(reduce='sum')
        last_model = None

        for e in range(epochs):  
            for idx in range(0, len(self._videos)):
                loss = 0
                model.train()
                optimizer.zero_grad()

                # Load features
                video = self._videos[idx]
                file = video.name
                feats = self._load_original_features(features_path, file)

                # Our batch size is 1, there is no padding and no need to mask
                # Change it if needed
                mask = torch.ones_like(feats)
                # Reconstructed features, predicted labels, commitment loss, distances, encoder output
                reconstructed_feats, _, commit_loss, _, _ = model(feats, mask, return_enc=True)

                ### Loss computation ###
                reconstruction_loss = 0
                # Compute reconstruction loss for each stage of MS-TCN
                for i in range(reconstructed_feats.shape[0]):
                    reconstruction_loss += loss_rec_fn(reconstructed_feats[i].unsqueeze(0), feats)

                loss += opt.vqt_commit_weight * commit_loss
                loss += opt.vqt_rec_weight * reconstruction_loss  

                loss.backward()
                optimizer.step() 

                # We take the last model as best model
                last_model = model.state_dict()    
            
            if opt.use_scheduler:
                scheduler.step()
            logger.debug(f'VQ - epoch {e}: total {loss.item():.4f}, commit {commit_loss.item():.4f}, rec {reconstruction_loss.item():.4f}')  
            

        # Load the last model and store embeddings for clustering 
        model.load_state_dict(last_model)
        self.vq_model_trained = model
        self.vq_optim = optimizer
        model.eval()
        self._features = None
        for video in self._videos:
            file = video.name
            feats = self._load_original_features(features_path, file)

            mask = torch.ones_like(feats)
            emb, _, = model.encode(feats, mask)

            if opt.force_update_z:
                labels = model.get_labels(feats, mask)
                video._z = labels.squeeze().detach().cpu().numpy()

            video._embedding = emb.squeeze().T.detach().cpu().numpy()
            video._adjust_embed_dim()

            start = 0 if self._features is None else self._features.shape[0]
            video.global_start = start
            self._features = join_data(self._features, video._embedding, np.vstack)
                
        self._embedded_feat = self._features
        self._total_fg_mask = np.ones(self._features.shape[0]).astype(bool)

        for video in self._videos:    
            video.update_indexes(self._features.shape[0])   

    def update_decoding(self):
        if opt.bg:
            for video in self._videos:
                video._z[video._valid_decoding] = -1
                             
    def update_likelihood(self):
        '''
        Likelihood for videos is computed using the trained VQ model.
        The likelihood of every frame to belong to a certain class is
        computed as the distance between the frame and the corresponding
        prototype.
        '''
        model = self.vq_model_trained
        model.eval()
        features_path = os.path.join(opt.original_feat_path, self._subaction) 
        
        for video in self._videos:
            file = video.name
            feats = self._load_original_features(features_path, file)
                
            mask = torch.ones_like(feats)
            distances = model.get_distances_sum(feats, mask)
            video._likelihood_grid = distances.squeeze().detach().cpu().numpy()

    def bg_update(self):
        if opt.bg:
            scores = None
            for video in self._videos:
                scores = join_data(scores, video.get_likelihood(), np.vstack)

            bg_trh_score = np.sort(scores, axis=0)[int((opt.bg_trh / 100) * scores.shape[0])]

            trh_set = []
            for action_idx in range(self._K):
                trh_set.append(bg_trh_score[action_idx])
            
            for video in self._videos:
                video.valid_likelihood_update(trh_set)    
 
    def _update_video_features(self, video_idx):
        '''
        Update the features of the video using the trained classifier.
        Used only when opt.use_cls and opt.update_w_cls_emb are set to True.
        '''
        video = self._videos[video_idx]
        if opt.load_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
       
        self._classifier.eval()
        embed = self._classifier.embed(torch.FloatTensor(features).to(opt.device)).cpu().detach().numpy()
        self._classifier.train()
        
        video._features = embed

    def _video_likelihood_grid(self, video_idx):
        video = self._videos[video_idx]
        if opt.load_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
       
        self._classifier.eval()
        scores = self._classifier(torch.FloatTensor(features).to(opt.device)).cpu().detach().numpy() # features).cuda()
        self._classifier.train()
        video._likelihood_grid = scores
        if opt.save_likelihood:
            video.save_likelihood()

    def vq_clustering(self):
        '''
        Provide labels for every frame using the trained VQ model.
        The label is computed as the closest prototype to the frame. 
        Similar to CTE, but with prototypes instead of KMeans.
        '''
        logger.debug('Clustering using HVQ')

        # If `train_vq` is not called, load the trained model from path
        if self.vq_model_trained is None:
            if opt.model_type == "single":
                model = SingleVQModel(num_stages=2, num_layers=10, num_f_maps=opt.f_maps, dim=opt.vq_dim, num_classes=self._K, latent_dim=opt.f_maps).to(opt.device)
            elif opt.model_type == "double":
                model = DoubleVQModel(num_stages=2, num_layers=10, num_f_maps=opt.f_maps, dim=opt.vq_dim, num_classes=self._K, latent_dim=opt.f_maps, ema_dead_code=opt.ema_dead_code).to(opt.device)
            elif opt.model_type == "triple":
                model = TripleVQModel(num_stages=2, num_layers=10, num_f_maps=opt.f_maps, dim=opt.vq_dim, num_classes=self._K, latent_dim=opt.f_maps, ema_dead_code=opt.ema_dead_code).to(opt.device)
            
            model.load_state_dict(torch.load(f'{opt.vq_model_path}/{self._subaction}_model'), strict=False)
        else:
            model = self.vq_model_trained

        self.vq_model = model
        self.vq_model.eval()
        # Get the labels (prototype indices) for every frame and likelihood grid (distance to the prototypes)
        model.eval()
        torch_feats = torch.from_numpy(self._embedded_feat[self._total_fg_mask]).transpose(0,1).unsqueeze(0).float().to(opt.device)
        original_cluster_labels, distances = model.get_labels_from_emb(torch_feats, torch.ones_like(torch_feats))
        distances = distances[0].detach().cpu().numpy()
        original_cluster_labels = original_cluster_labels[0].detach().cpu().numpy()

        # We exploit CTE way to order the clusters
        accuracy = Accuracy()
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_rt += list(video.temp)
            video._likelihood_grid = distances[video.global_range]

        long_rt = np.array(long_rt)

        cluster_labels = original_cluster_labels.copy()
        time2label = {}
        for label in np.unique(cluster_labels):
            cluster_mask = cluster_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        logger.debug('time ordering of labels')
        vq_to_order = {}
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            cluster_labels[original_cluster_labels == label] = time_idx
            vq_to_order[label] = time_idx

        shuffle_labels = np.arange(len(time2label))

        self.pseudo_gt_with_bg = np.ones(len(self._total_fg_mask)) * -1

        # use predefined by time order  for kmeans clustering
        self.pseudo_gt_with_bg[self._total_fg_mask] = cluster_labels

        logger.debug('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
        accuracy.predicted_labels = self.pseudo_gt_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass

        flat_gt2label = {key: val[0] for key, val in self._gt2label.items()}
        self.real_gt = np.array(long_gt)
        self.real_gt = np.vectorize(flat_gt2label.get)(self.real_gt)

        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        if opt.vis and opt.vis_mode != 'segm':
            dot_path = ''
            self.vis = Visual(mode=opt.vis_mode, save=True, svg=False, saved_dots=dot_path)
            
            data = self._embedded_feat
            
            self.vis.fit(data, long_gt, 'gt_', reset=False)
            self.vis.color(original_cluster_labels, 'kmean')
        ########################################################################

        logger.debug('Update video z for videos before GMM fitting')
        self.pseudo_gt_with_bg[self.pseudo_gt_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(self.pseudo_gt_with_bg[video.global_range])

        for video in self._videos:
            video.segmentation['cl'] = (video._z, self._label2gt)
            t = np.unique(cluster_labels[video.global_range[self._total_fg_mask]])
            self.buffer.add_sequence(features=cluster_labels[video.global_range[self._total_fg_mask]], framelabels=cluster_labels[video.global_range[self._total_fg_mask]], transcript=t)

        self.update_mean_lengths()
        self.update_prior()

        self.buffer.features = []
        self.buffer.transcript = []
        self.buffer.framelabels = []
        self.buffer.instance_counts = []
        self.buffer.label_counts = []
    
    def clustering(self):
        '''
        CTE clustering with KMeans
        '''
        logger.debug('.')
        np.random.seed(opt.seed)

        kmean = MiniBatchKMeans(n_clusters=self._K, random_state=opt.seed, batch_size=50)
        kmean.fit(self._embedded_feat[self._total_fg_mask])

        accuracy = Accuracy()
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video.gt)
            long_rt += list(video.temp)
        long_rt = np.array(long_rt)

        kmeans_labels = np.asarray(kmean.labels_).copy()
        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        logger.debug('time ordering of labels')
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            kmeans_labels[kmean.labels_ == label] = time_idx

        shuffle_labels = np.arange(len(time2label))

        self.pseudo_gt_with_bg = np.ones(len(self._total_fg_mask)) * -1

        # use predefined by time order  for kmeans clustering
        self.pseudo_gt_with_bg[self._total_fg_mask] = kmeans_labels

        logger.debug('Order of labels: %s %s' % (str(shuffle_labels), str(sorted(time2label))))
        accuracy.predicted_labels = self.pseudo_gt_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass

        flat_gt2label = {key: val[0] for key, val in self._gt2label.items()}
        self.real_gt = np.array(long_gt)
        self.real_gt = np.vectorize(flat_gt2label.get)(self.real_gt)

        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        if opt.vis and opt.vis_mode != 'segm':
            dot_path = ''
            self.vis = Visual(mode=opt.vis_mode, save=True, svg=False, saved_dots=dot_path)
            self.vis.fit(self._embedded_feat, long_gt, 'gt_', reset=False)
            self.vis.color(long_rt, 'time_')
            self.vis.color(kmean.labels_, 'kmean')
        ########################################################################

        logger.debug('Update video z for videos before GMM fitting')
        self.pseudo_gt_with_bg[self.pseudo_gt_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(self.pseudo_gt_with_bg[video.global_range])

        for video in self._videos:
            video.segmentation['cl'] = (video._z, self._label2gt)
            t = np.unique(kmeans_labels[video.global_range])
            self.buffer.add_sequence(features=kmeans_labels[video.global_range], framelabels=kmeans_labels[video.global_range], transcript=t)

        self.update_mean_lengths()
        self.update_prior()

        self.buffer.features = []
        self.buffer.transcript = []
        self.buffer.framelabels = []
        self.buffer.instance_counts = []
        self.buffer.label_counts = []
        
    def _count_subact(self):
        self._subact_counter = np.zeros(self._K)
        for video in self._videos:
            self._subact_counter += video.a

    def generate_pi(self, pi, n_ins=0, n_del=0):
        output = pi.copy()
        for _ in range(n_del):
            n = len(output)
            idx = np.random.randint(n)
            output.pop(idx)

        for _ in range(n_ins):
            m = len(pi)
            val = np.random.randint(m)
            n = len(output)
            idx = np.random.randint(n)
            output.insert(idx, val)

        return output
    
    @timing
    def fifa_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        max_score_list = []
        
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            if opt.bg:
                video.update_fg_mask()
            
            self.decoder.length_model = PoissonModel(self.mean_lengths)

            max_score, max_z, max_pi = self.fifa_video_decode(video, self.decoder)
            
            logger.debug('video length' + str(len(video._likelihood_grid)))
            max_score_list.append(max_score/len(video._likelihood_grid))
            
            if len(max_z) <= 0:
                continue

            self.pseudo_gt_with_bg[video.global_range] = max_z

            self.buffer.add_sequence(max_z[video.fg_mask], max_pi, max_z[video.fg_mask])
            self.update_prior()
            self.update_mean_lengths()
            

            video._subact_count_update()
            video._z = np.asarray(max_z).copy()

            name = str(video.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
            np.savetxt(join(opt.output_dir, 'segmentation', name), np.asarray(max_z), fmt='%d')     
            
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
            
        self._count_subact()

        logger.debug('Q value' + str(np.mean(max_score_list)))
        logger.debug(str(self._subact_counter))

    def fifa_video_decode(self, video, decoder):
        max_score = -np.inf
        max_z = []
        max_pi = []
        pi = video._pi    
        
        transcript = self.generate_pi(pi, n_ins=0, n_del=0)

        if np.sum(video.fg_mask):
            log_probs = video._likelihood_grid[video.fg_mask] 

            actions = torch.tensor(transcript, dtype=torch.long).unsqueeze(0).to(opt.device)
            durations = torch.tensor(video.a, dtype=torch.float32).unsqueeze(0).to(opt.device)
            
            priors = torch.from_numpy(self.mean_lengths / video.n_frames).to(opt.device)
            framewise_pred = torch.from_numpy(log_probs).transpose(0,1).unsqueeze(0).to(opt.device)
            
            duration_fifa, score = fifa(actions, durations, framewise_pred, priors=priors, num_epochs=50, transcript=transcript)
            pred_seg_expanded_dur = self.convert_segments_to_labels(actions, duration_fifa.detach(), video.n_frames)
            labels = pred_seg_expanded_dur[0].numpy()  

            z = np.ones(video.n_frames, dtype=int) * -1
            z[video.fg_mask] = labels[video.fg_mask]               
        else:
            z = np.ones(video.n_frames, dtype=int) * -1
            score = -np.inf

        if score > max_score:
            max_score = score
            max_z = z
            max_pi = transcript

        return max_score, max_z, max_pi
    
    def convert_segments_to_labels(self, action, duration, num_frames):
        assert  action.shape[0] == 1
        labels = action[0, :] 
        duration = duration[0, :]
        duration = duration / duration.sum()
        duration = (duration * num_frames).round().long()
        if duration.shape[0] == 0:
            duration = torch.tensor([num_frames])
            labels = torch.tensor([0])
        if duration.sum().item() != num_frames:
            # there may be small inconsistencies due to rounding.
            duration[-1] = num_frames - duration[:-1].sum()
        assert duration.sum().item() == num_frames, f"Prediction {duration.sum().item()} does not match number of frames {num_frames}."
        frame_wise_predictions = torch.zeros((1, num_frames))
        idx = 0
        for i in range(labels.shape[0]):
            frame_wise_predictions[0, idx:idx + duration[i]] = labels[i]
            idx += duration[i]
        return frame_wise_predictions

    @timing
    def cte_viterbi_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            if opt.bg:
                video.update_fg_mask()
                
            video.viterbi()
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
                
        self._count_subact()
        logger.debug(str(self._subact_counter))

    @timing
    def viterbi_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        max_score_list = []
        
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            if opt.bg:
                video.update_fg_mask()       
            
            self.decoder.length_model = PoissonModel(self.mean_lengths)

            max_score, max_z, max_pi = self.video_decode(video, self.decoder)
            
            logger.debug('video length' + str(len(video._likelihood_grid)))
            max_score_list.append(max_score/len(video._likelihood_grid))
            
            if len(max_z) <= 0:
                continue

            self.pseudo_gt_with_bg[video.global_range] = max_z

            self.buffer.add_sequence(max_z[video.fg_mask], max_pi, max_z[video.fg_mask])
            self.update_prior()
            self.update_mean_lengths()

            video._subact_count_update()
            video._z = np.asarray(max_z).copy()

            name = str(video.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
            np.savetxt(join(opt.output_dir, 'segmentation', name), np.asarray(max_z), fmt='%d')
            
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
        self._count_subact()
        
        logger.debug('Q value' + str(np.mean(max_score_list)))
        logger.debug(str(self._subact_counter))

    def video_decode(self, video, decoder):
        max_score = -np.inf
        max_z = []
        max_pi = []
        pi = video._pi     

        transcript = self.generate_pi(pi, n_ins=0, n_del=0)
        long_rt = video.temp

        kmeans_labels = video._z.copy() 
        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(long_rt[cluster_mask])
            time2label[r_time] = label

        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            kmeans_labels[video._z == label] = time_idx

        if np.sum(video.fg_mask):
            log_probs = video._likelihood_grid[video.fg_mask] - np.log(self.prior)
            log_probs = log_probs - np.max(log_probs) 
            decoder.grammar = SingleTranscriptGrammar(transcript, self._K)
            score, labels, segments = decoder.decode(log_probs)
            z = np.ones(video.n_frames, dtype=int) * -1
            z[video.fg_mask] = labels             
        else:
            z = np.ones(video.n_frames, dtype=int) * -1
            score = -np.inf
        
        if score > max_score:
            max_score = score
            max_z = z
            max_pi = transcript

        return max_score, max_z, max_pi


    def dummy_decoding(self):
        for video in self._videos:
            video._z = np.ones_like(video._z)

    def no_decoder(self):
        for video in self._videos:
            segments = torch.from_numpy(video._z).unsqueeze(0).to(opt.device)

    def without_temp_emed(self):
        logger.debug('No temporal embedding')
        self._embedded_feat = self._features.copy()

    def vis_seg(self, prediction, img_height = 20):
        colors = []
        for _ in range(2):
            colors.extend([[255, 0, 0], [0, 255, 0], [0, 0, 255],
                        [0, 255, 255], [255, 0, 255], [255, 255, 0],
                        [255, 192, 0], [0, 192, 255], [255, 0, 192],
                        [0, 255, 192], [192, 0, 255], [192, 255, 0],
                        [112, 48, 60], [60, 48, 112], [112, 60, 48],
                        [48, 60, 112], [48, 112, 60], [60, 112, 48],
                        [48, 0, 112], [48, 112, 0], [0, 112, 48],
                        [0, 48, 112], [112, 112, 0], [48, 48, 0]])
        colors = np.array(colors)


        img_width = len(prediction)
        segment_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for ind in range(segment_img.shape[1]):
            label = prediction[ind]
            if label == 0:
                segment_img[:, ind, :] = np.array([0, 0, 0])
            elif label != -1 and label <= 24:
                segment_img[:, ind, :] = colors[label - 1]
            elif label != -1 and label > 24:
                new_label = label % 17
                segment_img[:, ind, :] = colors[new_label - 1]
            elif label == -1:
                segment_img[:, ind, :] = np.array([255, 255, 255])
        return segment_img

    @timing
    def accuracy_corpus(self, prefix=''):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""
        accuracy = Accuracy()
        f1_score = F1Score(K=self._K, n_videos=len(self._videos))
        long_gt = []
        long_pr = []
        long_rel_time = []
        self.return_stat = {}

        for video in self._videos:
            long_gt += list(video.gt)
            long_pr += list(video._z)
            try:
                long_rel_time += list(video.temp)
            except AttributeError:
                pass
                # logger.debug('no poses')

        accuracy.gt_labels = long_gt
        accuracy.predicted_labels = long_pr
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]

        old_mof, total_fr = accuracy.mof(old_gt2label=self._gt2label)
        if not os.path.exists('./images'):
            os.makedirs('./images')
        if not os.path.exists(f'./images/{self._subaction}'):
            os.makedirs(f'./images/{self._subaction}')
        for video in self._videos:
            if prefix == 'final':
                fig = plt.figure()
                plt.title(video.name)
                plt.imshow(self.vis_seg(video._z))
                plt.axis('off')
                plt.savefig(f'./images/{self._subaction}/{video.name}_segm')
                plt.tight_layout()
                plt.close()
        self._gt2label = accuracy._gt2cluster
        self._label2gt = {}
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass
        acc_cur = accuracy.mof_val()
        logger.debug('%sAction: %s' % (prefix, self._subaction))
        logger.debug('%sMoF val: ' % prefix + str(acc_cur))
        logger.debug('%sprevious dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

        accuracy.mof_classes()
        accuracy.iou_classes()

        self.return_stat = accuracy.stat()

        f1_score.set_gt(long_gt)
        f1_score.set_pr(long_pr)
        f1_score.set_gt2pr(self._gt2label)
        if opt.bg:
            f1_score.set_exclude(-1)
        f1_score.f1()

        for key, val in f1_score.stat().items():
            self.return_stat[key] = val

        for video in self._videos:
            video.segmentation[video.iter] = (video._z, self._label2gt)
            
        from hvq.eval_utils.fully_f1 import f_score
        f1s = []
        for video in self._videos:
            tp, fp, fn = f_score(video.segmentation[0][0], video.gt, 0.5)
            prec = tp / (tp+fp)
            rec = tp / (tp+fn)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-13)
            f1s.append(f1)
        print(f"F1 overlap {np.mean(f1s)}")
        gt2lab = {k:v[0] for k,v in self._gt2label.items()}
        long_gt_t = np.vectorize(gt2lab.get)(long_gt)
        for ov in [0.1, 0.25, 0.5]:
            tp, fp, fn = f_score(long_pr, long_gt_t, ov)
            prec = tp / (tp+fp)
            rec = tp / (tp+fn)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-13)
            print(f"F1@{ov}: {f1}")
            
        print()

        if opt.vis:
            ########################################################################
            # VISUALISATION

            if opt.vis_mode != 'segm':
                long_pr = [self._label2gt[i] for i in long_pr]

                if self.vis is None:
                    self.vis = Visual(mode=opt.vis_mode, save=True, reduce=None)
                    self.vis.fit(self._embedded_feat, long_pr, 'iter_%d' % self.iter)
                else:
                    reset = prefix == 'final'
                    self.vis.color(labels=long_pr, prefix='iter_%d' % self.iter, reset=reset)
            else:
                ####################################################################
                # visualisation of segmentation
                if prefix == 'final':
                    colors = {}
                    cmap = plt.get_cmap('tab20')
                    for label_idx, label in enumerate(np.unique(long_gt)):
                        if label == -1:
                            colors[label] = (0, 0, 0)
                        else:
                            # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                            colors[label] = cmap(label_idx / len(np.unique(long_gt)))

                    dir_check(os.path.join(opt.dataset_root, 'plots'))
                    dir_check(os.path.join(opt.dataset_root, 'plots', opt.subaction))
                    fold_path = os.path.join(opt.dataset_root, 'plots', opt.subaction, 'segmentation')
                    dir_check(fold_path)
                    for video in self._videos:
                        if "P03" in video.name:
                            print()
                        path = os.path.join(fold_path, video.name + '.png')
                        name = video.name.split('_')
                        name = '_'.join(name[-2:])
                        plot_segm(path, video.segmentation, colors, name=name)
                ####################################################################
            ####################################################################

        return accuracy.frames()

    def resume_segmentation(self):
        logger.debug('resume precomputed segmentation')
        for video in self._videos:
            video.iter = self.iter
            video.resume()
        self._count_subact()

    def save_embed_feat(self):
        dir_check(ops.join(opt.data, 'embed'))
        dir_check(ops.join(opt.data, 'embed', opt.subaction))
        for video in self._videos:
            video_features = self._embedded_feat[video.global_range]
            feat_name = opt.resume_str + '_%s' % video.name
            np.savetxt(ops.join(opt.data, 'embed', opt.subaction, feat_name), video_features)

    def update_mean_lengths(self):
        self.mean_lengths = np.zeros( (self._K), dtype=np.float32 )
        for label_count in self.buffer.label_counts:
            self.mean_lengths += label_count
        instances = np.zeros((self._K), dtype=np.float32)
        for instance_count in self.buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 \
                else sum(self.mean_lengths) / sum(instances) for i in range(self._K) ] )

    def update_prior(self):
        # count labels
        self.prior = np.zeros((self._K), dtype=np.float32)
        for label_count in self.buffer.label_counts:
            self.prior += label_count
        self.prior = self.prior / np.sum(self.prior)
        # backup to uniform probability for unseen classes
        n_unseen = sum(self.prior == 0)
        self.prior = self.prior * (1.0 - float(n_unseen) / self._K)
        self.prior = np.array( [ self.prior[i] if self.prior[i] > 0 else 1.0 / self._K for i in range(self._K) ] )

    def plot_embs(self):
        from sklearn.decomposition import PCA
        plt.figure()
        for video in self._videos:
            diff = np.diff(video.gt)
            chan = [0] + (np.where(diff != 0)[0] + 1).tolist() + [video.n_frames-1]

            pca = PCA(2)
            nemb = video._embedding
            pca.fit(nemb)

            colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 0: 'black'}
            for i in range(len(chan)-1):
                s, e = chan[i], chan[i+1]                
                ab = pca.transform(nemb[s:e]).T
                plt.scatter(ab[0,:], ab[1,:], c=colors[video.gt[s+1]])

        plt.show()
        print()

    def update_corpus(self):

        self._acc_old = 0
        self._videos = []
        # init with ones for consistency with first measurement of MoF
        self._subact_counter = np.ones(self._K)
        self._gaussians = {}
        self._inv_count_stat = np.zeros(self._K)
        self._embedding = None
        self._gt2label = None
        self._label2gt = {}

        self._with_bg = opt.bg
        self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features = None
        self._embedded_feat = None
        self._init_videos()