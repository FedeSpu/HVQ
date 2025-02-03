'''
The VQ model implementation is based on the following codes:
- https://github.com/lucidrains/vector-quantize-pytorch
- https://github.com/GoGoDuck912/pytorch-vector-quantization/tree/main
'''
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

from sklearn.cluster import KMeans
from ute.utils.arg_pars import opt

__author__ = 'Federico Spurio'
__date__ = 'February 2025'

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, eps=1e-5, threshold_ema_dead_code=2, kmeans=False, cos_sim=False, first=False):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.cos_sim = cos_sim
        self.kmeans = kmeans
        
        # Disclaimer: First dimension = 1 is the number of codebooks, not the batch size
        self._embedding = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))
        self._codebook = self._embedding
        self._commitment_cost = commitment_cost

        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        self.first = first

        self.register_buffer('initted', torch.Tensor([False]))
        # Disclaimer: First dimension same as comment above
        self.register_buffer('cluster_size', torch.zeros(1, num_embeddings))
        self.register_buffer('embed_avg', self._embedding.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
    
        if self.kmeans:
            km = KMeans(n_clusters=self._num_embeddings, n_init=10, random_state=42)
            # FIXME: not batched, only 1 batch
            km.fit(data[0].detach().cpu())
            embed = torch.from_numpy(km.cluster_centers_).unsqueeze(0)
        else:   
            embed = torch.empty((1, self._num_embeddings, data.shape[2]))
            nn.init.kaiming_uniform_(embed)   

        self._embedding.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = batched_sample_vectors(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self._embedding.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        
        ## Flatten input, for batch size > 1
        # flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = inputs

        self.init_embed_(flat_input)
        
        # Calculate distances
        if self.cos_sim:
            flat_input = l2norm(flat_input)
            embed = l2norm(self._embedding)
            # Cosine similarity (not distance)
            distances = einsum('h n d, h c d -> h n c', flat_input, embed)
        else:
            distances = -torch.cdist(flat_input, self._embedding)
            
        # Encoding
        encoding_indices = torch.argmax(distances, dim=2) 

        encoding_onehot = F.one_hot(encoding_indices, self._num_embeddings).type(torch.float32)
        if opt.gumbel:
            encoding_indices, encoding_onehot = gumbel_sample(distances, dim=-1, stochastic=True, training=self.training, temperature=0.9)

        if self.training:
            if self.cos_sim:
                bins = encoding_onehot.sum(dim = 1)

                self.cluster_size.data.lerp_(bins, 1 - self.decay)

                zero_mask = (bins == 0)
                bins = bins.masked_fill(zero_mask, 1.)

                embed_sum = einsum('h n d, h n c -> h c d', flat_input, encoding_onehot)

                embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
                embed_normalized = l2norm(embed_normalized)

                embed_normalized = torch.where(
                    rearrange(zero_mask, '... -> ... 1'),
                    embed,
                    embed_normalized
                )

                if opt.use_ema:
                    self._embedding.data.lerp_(embed_normalized, 1 - self.decay)
                self.expire_codes_(inputs)
            else:
                cluster_size = encoding_onehot.sum(dim = 1)

                self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

                embed_sum = einsum('h n d, h n c -> h c d', flat_input, encoding_onehot)
                self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)

                cluster_size = laplace_smoothing(self.cluster_size, self._num_embeddings, self.eps) * self.cluster_size.sum()

                embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
                # EMA update
                self._embedding.data.copy_(embed_normalized)
                self.expire_codes_(inputs)
        
        # Quantize and unflatten
        quantized = batched_embedding(encoding_indices.unsqueeze(0), self._embedding)[0]
        # Loss
        commit_loss = F.mse_loss(quantized.detach(), inputs)     
        
        if opt.vq_loss:
            vq_loss = F.mse_loss(quantized, inputs.detach())
        else:
            vq_loss = 0
        loss = vq_loss + self._commitment_cost * commit_loss

        if opt.dist_loss and not self.first:
            loss += opt.dist_weight * torch.max(torch.tensor(0., device=embed.device), 
                                                (einsum('h n d, h c d -> h n c', embed, embed).squeeze() - torch.eye(embed.size(1), device=embed.device)).mean()-opt.dist_margin)
    
        quantized = inputs + (quantized - inputs).detach()        
        # Convert quantized from BLC -> BCL
        return quantized.permute(0, 2, 1).contiguous(), encoding_indices, loss, distances 
    
    

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    straight_through = False,
    reinmax = False,
    dim = -1,
    training = True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), 'reinmax can only be turned on if using straight through gumbel softmax'

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim = dim)
        π1 = (one_hot + (logits / temperature).softmax(dim = dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim = 1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim = dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot