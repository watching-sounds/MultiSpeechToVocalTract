import torch
import torch.nn as nn

__all__ = [
    'GRUWrapper'
]

class GRUWrapper(nn.Module):
    """
    Alternate GRU network wrapper for sequence modeling.
    """
    def __init__(self, in_feats, hid_feats, out_feats, num_layers,
                 drop_prob=0.2, use_bidir=False, embed_size=0):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.use_bidir = use_bidir
        self.dir_mult = 2 if use_bidir else 1

        self.input_proj = nn.Linear(in_feats, embed_size) if embed_size > 0 else None
        input_to_gru = embed_size if embed_size > 0 else in_feats

        self.core = nn.GRU(
            input_size=input_to_gru,
            hidden_size=hid_feats,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob,
            bidirectional=use_bidir
        )
        self.activation = nn.ReLU()
        self.output_proj = nn.Linear(hid_feats * self.dir_mult, out_feats)

        self._config = dict(
            in_feats=self.in_feats,
            hid_feats=self.hid_feats,
            out_feats=self.out_feats,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            drop_prob=self.drop_prob,
            use_bidir=self.use_bidir
        )

    def forward(self, seq, h_init):
        if self.input_proj is not None:
            seq = self.input_proj(seq)
        rnn_out, h_last = self.core(seq, h_init)
        logits = self.output_proj(self.activation(rnn_out))
        return logits, h_last

    def predict(self, seq):
        h0 = self._zero_init_hidden(seq.size(0))
        preds, _ = self.forward(seq, h0)
        return preds

    def _zero_init_hidden(self, batch_sz):
        param = next(self.parameters())
        return param.new_zeros(
            self.num_layers * self.dir_mult,
            batch_sz,
            self.hid_feats
        )

    def store(self, path):
        torch.save({
            'config': self._config,
            'weights': self.state_dict()
        }, path)

    @staticmethod
    def restore(path, device):
        checkpoint = torch.load(path, map_location=device)
        model = GRUWrapper(**checkpoint['config'])
        model.load_state_dict(checkpoint['weights'])
        model.to(device)
        return model
