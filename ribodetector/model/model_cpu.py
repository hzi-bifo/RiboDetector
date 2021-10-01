import torch
from torch import jit
import torch.nn as nn
from torch import Tensor
from ribodetector.base import BaseModel


class SeqModel(BaseModel):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 num_classes,
                 batch_first=True,
                 bidirectional=True,
                 pack_seq=False):
        super(SeqModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,      # width of input
            hidden_size=hidden_size,     # number of rnn hidden unit
            num_layers=num_layers,       # number of RNN layers
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size * 2, num_classes)    # output layer

        self.forward = self.forward_last

    def forward_last(self, x):
        r_out, _ = self.rnn(x, None)

        # last_out = r_out[torch.arange(
        #     x.size(0)), (x.sum(2) != 0).cumsum(1).argmax(1), :]

        last_out = last_out_items(x, r_out)
        out = self.out(last_out)
        return out

    # If there is no ambigous nucleotide in the sequence
    def forward_fast(self, x):
        r_out, _ = self.rnn(x, None)

        last_out = r_out[torch.arange(
            x.size(0)), x.sum((1, 2)).long() - 1, :]
        out = self.out(last_out)
        return out

    def forward_lastforward_firstreverse(self, x):
        r_out, (h_n, _c_n) = self.rnn(x, None)

        last_out = lastforward_firstreverse_out_items(
            x, r_out, h_n, self.hidden_size)
        out = self.out(last_out)
        return out


@jit.script
def last_out_items(ts: Tensor, r_out: Tensor) -> Tensor:
    last_out = r_out[torch.arange(ts.size(0)), ts.size(
        1) - 1 - ts.sum(2).flip(1).argmax(1), :]

    return last_out


# The last forward unpad and first reverse rnn out or reverse hn
@jit.script
def lastforward_firstreverse_out_items(ts: Tensor,
                                       r_out: Tensor,
                                       h_n: Tensor,
                                       hidden_size: int) -> Tensor:
    last_out = torch.cat((r_out[torch.arange(ts.size(0)), ts.size(
        1) - 1 - ts.sum(2).flip(1).argmax(1), :hidden_size], h_n[1]), 1)

    return last_out
