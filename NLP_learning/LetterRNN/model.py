import torch
from torch import nn
import torch.nn.functional as F
from constant import EMBEDDING_LENGTH

class RNN(nn.Module):
    def __init__(self, hidden_utils=64, embedding_dim=64, dropout_rate=0.2):
        super().__init__()
        self.hidden_utils = hidden_utils
        self.drop = nn.Dropout(dropout_rate) # normalize
        self.encoder = nn.Embedding(EMBEDDING_LENGTH, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_utils, batch_first=True)
        self.decoder = nn.Linear(hidden_utils, EMBEDDING_LENGTH)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, word:torch.Tensor):
        # word/x shape:[batch, max_word_length]([B, T])
        batch, Tx = word.shape[0:2]
        first_letter = word.new_zeros(batch, 1) # add 0 to the beginning as the flag of start([B, 1])
        x = torch.cat([first_letter, word[:, 0:-1]], dim=1) # [B, 1+T-1] -> [B, T]
        hidden = torch.zeros(1, batch, self.hidden_utils, device=word.device) # a

        emb = self.drop(self.encoder(x)) # replace every index to embedding_dim([B, T] -> [B, T, hidden_units])
        output, hidden = self.rnn(emb, hidden) # [B, T, ~], [1, B, ~]
        y = self.decoder(output.reshape(batch * Tx, -1)) # Linear prefer 2_dim input(-> [B * T, ~] -> [B * T, EMBEDDING])

        return y.reshape(batch, Tx, -1) # recover the shape([B, T, EMBEDDING])

    @torch.no_grad()
    def language_model(self, word:torch.Tensor): # word[word, letter]
        batch, Tx = word.shape[0:2]
        y_hat = self.forward(word)
        y_hat = F.softmax(y_hat, 2) # normalize according to dim2(embedding)
        output = torch.ones(batch, device=word.device)
        for i in range(Tx):
            # torch.arange(batch) generate a sequence like [0, 1, 2, ..., batch-1],
            # will output the shape of [2], rather than [2, 2] when using ':'
            # i represent the current letter (ith)
            # tensor shape of [batch], containing the index of letter of the ith place
            probs = y_hat[torch.arange(batch), i, word[:, i]]
            output *= probs # P(w_1)P(w_2|w_1)...
        return output