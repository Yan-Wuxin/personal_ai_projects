import torch
from torch import nn

class BiLSTM_Sentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch_size, sentence_len]
        embedded = self.dropout(self.embedding(text))
        # RNN will return the whole sequence
        # output: [batch_size, sentence_len, hid_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hid_dim]
        output, hidden = self.rnn(embedded)
        cat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) # dim = 1 means concatenating in horizonal direction
        return self.fc(self.dropout(cat_hidden))