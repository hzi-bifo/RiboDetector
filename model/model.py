import torch
from torch import jit
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from base import BaseModel
from torch.nn.utils.rnn import PackedSequence


class SeqModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 batch_first=True,
                 bidirectional=True,
                 pack_seq=True):
        super(SeqModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,      # width of input
            hidden_size=hidden_size,     # number of rnn hidden unit
            num_layers=num_layers,       # number of RNN layers
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

        self.out = nn.Linear(hidden_size * 2, num_classes)    # output layer

        if pack_seq:
            self.forward = self.forward1
        else:
            self.forward = self.forward2

    # packedsequence input, extract last unpad output
    def forward1(self, x):
        r_out, _ = self.rnn(x, None)
        last_out = last_items(pack=r_out, unsort=True)

        out = self.out(last_out)
        return out

    # paddedsequence input, extract last unpad output
    def forward2(self, x):
        r_out, _ = self.rnn(x, None)
        # last_out = r_out[torch.arange(
        #     x.size(0)), (x.sum(2) != 0).cumsum(1).argmax(1), :]
        # last_out = r_out[torch.arange(x.size(0)), x.size(
        #     1) - 1 - (torch.flip(x.sum(2), [1])).argmax(1), :]

        last_out = last_pad_out_items(x, r_out)

        out = self.out(last_out)
        return out

    # paddedsequence input, extract last output no matter if they are from padded step
    def forward3(self, x):
        r_out, _ = self.rnn(x, None)

        out = self.out(r_out[:, -1, :])
        return out

    # paddedsequence or packedsequence input, extract h_n output
    def forward4(self, x):
        _r_out, (h_n, _c_n) = self.rnn(x, None)
        last_out = torch.cat((h_n[0], h_n[1]), 1)
        out = self.out(last_out)
        return out


@jit.script
def last_pad_out_items(ts: Tensor, r_out: Tensor) -> Tensor:
    last_out = r_out[torch.arange(ts.size(0)), ts.size(
        1) - 1 - ts.sum(2).flip(1).argmax(1), :]

    return last_out


@jit.script
def sorted_lengths(pack: PackedSequence) -> Tuple[Tensor, Tensor]:
    indices = torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )
    lengths = ((indices + 1)[:, None] <=
               pack.batch_sizes[None, :]).long().sum(dim=1)
    return lengths, indices


@jit.script
def sorted_first_indices(pack: PackedSequence) -> Tensor:
    return torch.arange(
        pack.batch_sizes[0],
        dtype=pack.batch_sizes.dtype,
        device=pack.batch_sizes.device,
    )


@jit.script
def sorted_last_indices(pack: PackedSequence) -> Tensor:
    lengths, indices = sorted_lengths(pack)
    cum_batch_sizes = torch.cat([
        pack.batch_sizes.new_zeros((2,)),
        torch.cumsum(pack.batch_sizes, dim=0),
    ], dim=0)
    return cum_batch_sizes[lengths] + indices


@jit.script
def first_items(pack: PackedSequence, unsort: bool) -> Tensor:
    if unsort and pack.unsorted_indices is not None:
        return pack.data[pack.unsorted_indices]
    else:
        return pack.data[:pack.batch_sizes[0]]


@jit.script
def last_items(pack: PackedSequence, unsort: bool) -> Tensor:
    indices = sorted_last_indices(pack=pack)
    if unsort and pack.unsorted_indices is not None:
        indices = indices[pack.unsorted_indices]
    return pack.data[indices]
