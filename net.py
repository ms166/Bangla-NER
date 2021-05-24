import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from dataset import VOCAB
from crf import CRF

class Net(nn.Module):
    def __init__(self, top_rnns=True, vocab_size=None, device='cpu', finetuning=True):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, dropout=0.5, input_size=768, hidden_size=768//2, batch_first=True)
            self.fc = nn.Sequential(
                  nn.Linear(768, 512),
                  nn.Dropout(0.5),
                  nn.Linear(512, vocab_size)
            )

        self.device = device
        self.finetuning = finetuning
        self.crf = CRF(len(VOCAB))

    def forward(self, x, y=None, epoch=None):

        x=x.to(self.device)
        if y is not None:
            y=y.to(self.device)

        if self.training and self.finetuning and (epoch>5):
            self.xlmr.train()
            enc = self.xlmr(x).last_hidden_state
        else:
            self.xlmr.eval()
            with torch.no_grad():
                enc = self.xlmr(x).last_hidden_state

        if self.top_rnns:
            enc, _ = self.rnn(enc)
        logits_focal = self.fc(enc)
        y_hat = self.crf.forward(logits_focal)
        logits_crf = self.crf.loss(logits_focal, y)
        return logits_focal, logits_crf, y, y_hat
