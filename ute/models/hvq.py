'''
Single stages tcn are based on https://github.com/yabufarha/ms-tcn
'''
import torch
from torch import nn
import torch.nn.functional as F
import copy
import random

from ute.models.vqlight import VectorQuantizer
from ute.utils.arg_pars import opt

random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
    
__author__ = 'Federico Spurio'
__date__ = 'February 2025'

class SingleStageDecoder(nn.Module):
    # num_classes is kept for consistency with the encoder
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageDecoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size)
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)

        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]    
    

class SingleStageEncoder(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageEncoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size)
                )
                for i in range(num_layers)
            ]
        )

        self.class_pred = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask, class_p=False):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)

        if class_p:
            pred_class = self.class_pred(out)
            return out, F.softmax(pred_class, dim=1)
            
        return out



class SingleVQModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, latent_dim):
        super(SingleVQModel, self).__init__()
        self.stage1 = SingleStageEncoder(num_layers, num_f_maps, dim, num_classes, 3)
        self.stages_enc = nn.ModuleList([copy.deepcopy(SingleStageEncoder(num_layers, num_f_maps, num_f_maps, num_classes, 3)) for s in range(num_stages-1)])
        
        self.vq = VectorQuantizer(
            num_embeddings=num_classes,
            embedding_dim=latent_dim,
            commitment_cost=1.,
            decay=0.8,
            threshold_ema_dead_code=opt.ema_dead_code,
            kmeans=True,
            cos_sim=True
        )
        
        self.stages_dec = nn.ModuleList([copy.deepcopy(SingleStageDecoder(num_layers, num_f_maps, num_f_maps, num_classes, 3)) for s in range(num_stages)])
        self.conv_out = nn.Conv1d(num_f_maps, dim, 1)

    def encode(self, x, mask):
        out = self.stage1(x, mask)
        output = out.unsqueeze(0)
        
        for se in self.stages_enc:
            out, cp = se(out * mask[:, 0:1, :], mask, class_p=True)  
            output = torch.cat((output, out.unsqueeze(0)), dim=0)            

        return out, output
    
    def get_distances_sum(self, x, mask):
        enc, _ = self.encode(x, mask)
        _, _, _, d1 = self.vq(enc)
        
        return d1
    
    def get_labels(self, x, mask, get_dist=False):
        enc, _ = self.encode(x, mask)
        _, indices, _, distances = self.vq(enc)
        
        if get_dist:
            return indices, distances
        return indices
    
    def get_labels_from_emb(self, x, mask):
        _, indices, _, distances = self.vq(x)

        return indices, nn.functional.log_softmax(distances, dim=2)

    def forward(self, x, mask, return_enc=False):
        enc, output_all = self.encode(x, mask)

        quantized, _, loss_commit, distances = self.vq(enc)

        out = quantized
        for sd in self.stages_dec:
            out = sd(out, mask)

        out = self.conv_out(out) * mask[:,0:1,:]

        if return_enc:
            return out, output_all, loss_commit, F.softmax(distances, dim=2), enc
        return out, output_all, loss_commit, F.softmax(distances, dim=2), enc

    
    
def map_tensors(i1, i2):
    i1 = i1.clone().squeeze()
    i2 = i2.clone().squeeze()
    
    mapping_dict = {}
    for val in torch.unique(i2):
        # Find indices where i2 equals the current unique value
        indices = torch.where(i2 == val)[0]
        # Find corresponding values in i1
        i1_vals = i1[indices.tolist()]
        # Get unique values to avoid repetitions
        unique_i1_vals = torch.unique(i1_vals).tolist()
        # Store in dictionary, directly if single value, or as a list if multiple values
        mapping_dict[val.item()] = unique_i1_vals[0] if len(unique_i1_vals) == 1 else unique_i1_vals
    
    return mapping_dict

class DoubleVQModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, latent_dim, ema_dead_code):
        super(DoubleVQModel, self).__init__()
        #### Encoder ####
        self.stage1 = SingleStageEncoder(num_layers, num_f_maps, dim, num_classes, 3)
        self.stages_enc = nn.ModuleList([copy.deepcopy(SingleStageEncoder(num_layers, num_f_maps, num_f_maps, num_classes, 3)) for s in range(num_stages-1)])
        
        #### Vector Quantizers ####
        self.vq = VectorQuantizer(
            num_embeddings=num_classes*opt.vq_class_multiplier,
            embedding_dim=latent_dim,
            commitment_cost=1.,
            decay=opt.vq_decay,
            threshold_ema_dead_code=ema_dead_code,
            kmeans=opt.vq_kmeans,
            cos_sim=True,
            first=True
        )
        
        self.vq2 = VectorQuantizer(
            num_embeddings=num_classes,
            embedding_dim=latent_dim,
            commitment_cost=1.,
            decay=opt.vq_decay,
            threshold_ema_dead_code=1,  
            kmeans=opt.vq_kmeans,
            cos_sim=True
        )
        
        #### Decoder ####
        self.stages_dec = nn.ModuleList([copy.deepcopy(SingleStageDecoder(num_layers, num_f_maps, num_f_maps, num_classes, 3)) for s in range(num_stages)])
        self.conv_out = nn.Conv1d(num_f_maps, dim, 1)

    def encode(self, x, mask):
        out = self.stage1(x, mask)
        output = out.unsqueeze(0)
        for se in self.stages_enc:
            out = se(out * mask[:, 0:1, :], mask)
            output = torch.cat((output, out.unsqueeze(0)), dim=0)

        return out, output
    
    def get_labels(self, x, mask):
        enc, _ = self.encode(x, mask)
        quantized1, i1, _, _ = self.vq(enc)
        _, indices, _, _ = self.vq2(quantized1)
        
        return indices
    
    def get_distances_sum(self, x, mask):
        enc, _ = self.encode(x, mask)
        quantized1, i1, _, d1 = self.vq(enc)
        _, i2, _, d2 = self.vq2(quantized1)
        
        map_indices = map_tensors(i1, i2)
        
        final_distances = d2.clone()
        for k, vals in map_indices.items():
            if type(vals) is int:
                final_distances[:,:,k] += d1[:,:,vals]
                continue
            for v in vals:
                final_distances[:,:,k] += d1[:,:,v]/len(vals)
                
        return nn.functional.log_softmax(final_distances, dim=2)
    
    def get_labels_from_emb(self, x, mask):
        quantized1, i1, _, d1 = self.vq(x)
        _, indices, _, d2 = self.vq2(quantized1)
        
        distances = d2 # d1 + d2

        return indices, nn.functional.log_softmax(distances, dim=2)

    def forward(self, x, mask, return_enc=False, image=False):
        # Project the input into the embedding space with encoder
        enc, output_all = self.encode(x, mask)

        # Quantize the embeddings in two steps
        quantized1, _, loss_commit1, _ = self.vq(enc)
        quantized, _, loss_commit2, distances = self.vq2(quantized1, image=image)
        
        # Commit loss is the sum of the two losses
        loss_commit = loss_commit1 + loss_commit2

        # Reconstruct the input
        out = quantized
        for sd in self.stages_dec:
            out = sd(out, mask)
        out = self.conv_out(out) * mask[:,0:1,:]

        if return_enc:
            return out, output_all, loss_commit, F.softmax(distances, dim=2), enc
        return out, output_all, loss_commit, F.softmax(distances, dim=2)
      

class TripleVQModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, latent_dim, ema_dead_code):
        super(TripleVQModel, self).__init__()
        self.stage1 = SingleStageEncoder(num_layers, num_f_maps, dim, num_classes, 3)
        self.stages_enc = nn.ModuleList([copy.deepcopy(SingleStageEncoder(num_layers, num_f_maps, num_f_maps, num_classes, 3)) for s in range(num_stages-1)])
        
        self.vq = VectorQuantizer(
            num_embeddings=num_classes*opt.vq_class_multiplier1,
            embedding_dim=latent_dim,
            commitment_cost=1.,
            decay=0.8,
            threshold_ema_dead_code=ema_dead_code,
            kmeans=True,
            cos_sim=True
        )
        
        self.vq2 = VectorQuantizer(
            num_embeddings=num_classes*opt.vq_class_multiplier2,
            embedding_dim=latent_dim,
            commitment_cost=1.,
            decay=0.8,
            threshold_ema_dead_code=1,
            kmeans=True,
            cos_sim=True
        )

        self.vq3 = VectorQuantizer(
            num_embeddings=num_classes,
            embedding_dim=latent_dim,
            commitment_cost=1.,
            decay=0.8,
            threshold_ema_dead_code=1,
            kmeans=True,
            cos_sim=True
        )
        
        self.stages_dec = nn.ModuleList([copy.deepcopy(SingleStageDecoder(num_layers, num_f_maps, num_f_maps, num_classes, 3)) for s in range(num_stages)])
        self.conv_out = nn.Conv1d(num_f_maps, dim, 1)

    def encode(self, x, mask):
        out = self.stage1(x, mask)
        output = out.unsqueeze(0)
        for se in self.stages_enc:
            out = se(out * mask[:, 0:1, :], mask)
            output = torch.cat((output, out.unsqueeze(0)), dim=0)

        return out, output
    
    def get_labels(self, x, mask):
        enc, _ = self.encode(x, mask)
        quantized1, _, _, _ = self.vq(enc)
        quantized2, _, _, _ = self.vq2(quantized1)
        _, indices, _, _ = self.vq3(quantized2)
        
        return indices
    
    def get_distances_sum(self, x, mask):
        enc, _ = self.encode(x, mask)
        quantized1, i1, _, d1 = self.vq(enc)
        _, i2, _, d2 = self.vq2(quantized1)
        
        quantized1, i1, _, d1 = self.vq(enc)
        quantized2, i2, _, d2 = self.vq2(quantized1)
        _, i3, _, d3 = self.vq2(quantized2)
        
        map_indices = map_tensors(i1, i2)
        
        inter_distances = d2.clone()
        for k, vals in map_indices.items():
            if type(vals) is int:
                inter_distances[:,:,k] += d1[:,:,vals]
                continue
            for v in vals:
                inter_distances[:,:,k] += d1[:,:,v]/len(vals)
                
        map_indices = map_tensors(i2, i3)
        
        final_distances = d3.clone()
        for k, vals in map_indices.items():
            if type(vals) is int:
                final_distances[:,:,k] += inter_distances[:,:,vals]
                continue
            for v in vals:
                final_distances[:,:,k] += inter_distances[:,:,v]/len(vals)
        
        return final_distances

    def get_labels_from_emb(self, x, mask):
        quantized1, _, _, d1 = self.vq(x)
        quantized2, _, _, d2 = self.vq2(quantized1)
        _, indices, _, d3 = self.vq3(quantized2)
        
        distances = d3 # d1 + d2

        return indices, nn.functional.log_softmax(distances, dim=2)

    def forward(self, x, mask, return_enc=False, image=False):
        enc, output_all = self.encode(x, mask)

        quantized1, _, loss_commit1, _ = self.vq(enc)
        quantized2, _, loss_commit2, _ = self.vq2(quantized1)
        quantized, _, loss_commit3, distances = self.vq2(quantized2)
        
        loss_commit = loss_commit1 + loss_commit2 + loss_commit3

        out = quantized
        for sd in self.stages_dec:
            out = sd(out, mask)

        out = self.conv_out(out) * mask[:,0:1,:]

        if return_enc:
            return out, output_all, loss_commit, F.softmax(distances, dim=2), enc
        return out, output_all, loss_commit, F.softmax(distances, dim=2)
       

