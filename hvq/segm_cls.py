import torch
from torch import nn
import torch.nn.functional as F

from hvq.utils.arg_pars import opt
from hvq.utils.logging_setup import logger


class Segm_CLS(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Segm_CLS, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # LSTM layer
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=1)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def embed(self, x):
        # out, _ = self.lstm(x)
        out, _ = self.gru(x)
        return out

    def forward(self, x):
        # _, (hn, cn) = self.lstm(x)  # hn of shape (D*num_layers, Batch, Hout)
        _, hn= self.gru(x)  # hn of shape (D*num_layers, Batch, Hout)
        if hn.shape[0] > 1:
            hn = hn[-1].unsqueeze(0)

        # Index hidden state of last time step
        out = self.fc(hn.permute(1,0,2))
        return F.softmax(out, dim=2)
    

def create_model(K):
    torch.manual_seed(opt.seed)
    model = Segm_CLS(opt.embed_dim, opt.embed_dim*2, K).to(opt.device)
    loss = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr*10, weight_decay=opt.weight_decay)

    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer