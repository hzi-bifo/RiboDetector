import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SeqModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 batch_first=True,
                 bidirectional=True):
        super(SeqModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,      # width of input
            hidden_size=hidden_size,     # number of rnn hidden unit
            num_layers=num_layers,       # number of RNN layers
            batch_first=batch_first,
            bidirectional=bidirectional,
            # dropout=DROPOUT
        )

        self.out = nn.Linear(hidden_size * 2, num_classes)    # output layer

    def forward(self, x):
        r_out, _ = self.rnn(x, None)
        # for i in r_out[:, -1, :]:
        #     hidden_fh.write('\t'.join(map(str, i.tolist())) + '\n')
        out = self.out(r_out[:, -1, :])
        return out
