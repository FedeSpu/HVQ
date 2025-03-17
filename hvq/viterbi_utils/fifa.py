#######################################
# code written by Nadine Behrmann
#######################################
import torch
from torch.autograd import Variable
from hvq.utils.arg_pars import opt


def get_p_matrix(framewise_pred, transcript):
    # returns matrix with framewise probabilities of size num_segments x seq_len
    framewise_pred = torch.softmax(framewise_pred, dim=1)
    P = []
    for i in range(transcript.shape[1]):
        P.append(framewise_pred[0, transcript[0, i], :])
    # add eps for stability
    P = - torch.log(torch.stack(P) + (1e-16))
    return P

def get_m_matrix(lengths, seq_len, sharpness, device):
    bn = torch.cumsum(lengths, dim=0)
    center = bn - lengths / 2
    width = lengths / 2
    t = torch.linspace(0, 1, seq_len).to(device)
    term1 = torch.exp(sharpness * (t[None, :] - center[:, None] - width[:, None])) + 1
    term2 = torch.exp(sharpness * (-t[None, :] + center[:, None] - width[:, None])) + 1
    M = 1 / (term1 * term2)
    return M

def fifa(action, duration=None, framewise_pred=None, priors=None, uniform_dur=False, num_epochs=10000, sharpness=None, lr=None, transcript=None):
    # get fixed probability matrix P[n, t] = probability that frame t belongs to segment n
    P = get_p_matrix(framewise_pred, action) # - 2)
    seq_len = framewise_pred.shape[2]
    
    sharpness = opt.fifa_sharpness; lr = opt.fifa_lr; num_epochs = opt.fifa_epochs

    means = priors
    if transcript:
        means = priors[transcript]
        transcript = torch.tensor(transcript, device=opt.device)
        duration = duration[:, transcript]

    if uniform_dur:
        duration = torch.ones_like(duration)
    duration = duration / duration.sum()

    log_length = torch.log(duration[0, :] + (1e-16))
    log_length = Variable(log_length, requires_grad=True)
    if opt.fifa_use_adam:
        optim = torch.optim.AdamW([log_length], lr=opt.fifa_lr)
    else:
        optim = DummyOpt()

    with torch.enable_grad():
        for epoch in range(num_epochs):
            optim.zero_grad()
            length = torch.exp(log_length)
            M = get_m_matrix(length, seq_len, sharpness, opt.device) 
            E_o = (P * M).mean()
            E_l = (length - means).abs().sum()
            
            loss = E_o + opt.fifa_weight * E_l + torch.abs(length.sum() - 1)
                            
            loss.backward()
            if not opt.fifa_use_adam:
                log_length.data -= lr * log_length.grad.data
                log_length.grad.data.zero_()
            optim.step()
    return torch.exp(log_length.data).unsqueeze(0), loss.item()

class DummyOpt():
    def __init__(self) -> None:
        pass
    
    def zero_grad(self):
        pass
    
    def step(self):
        pass