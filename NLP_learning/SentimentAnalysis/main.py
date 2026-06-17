import time
import intel_extension_for_pytorch as ipex
from collections import Counter
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from SentimentAnalysis.read_imdb import read_imdb
from model import BiLSTM_Sentiment
import pickle

# mapping table: collecting vocabularies from dataset, then generating embedding vectors
class Vocab:
    def __init__(self, texts, max_size = 25000):
        counter = Counter()
        # print(texts[0])
        # print(texts[1])
        for text in texts:
            counter.update(self.tokenizer(text))
        self.itos = ['<pad>', '<unk>'] + [w for w, _ in counter.most_common(max_size)]
        self.stoi = {w : i for i, w in enumerate(self.itos)}
        self.pad_idx = self.stoi['<pad>']
        self.unk_idx = self.stoi['<unk>']

    def tokenizer(self, text):
        return text.lower().split()

    def encode(self, text):
        return [self.stoi.get(w, self.unk_idx) for w in self.tokenizer(text)]


class IMDBDataset(Dataset):
    def __init__(self, is_train, vocab=None, dir='../data/aclImdb'):
        super().__init__()
        pos_lines = read_imdb(dir, 'pos', is_train)
        neg_lines = read_imdb(dir, 'neg', is_train)
        self.pos_len = len(pos_lines)
        self.neg_len = len(neg_lines)
        self.lines = pos_lines + neg_lines
        if vocab is None:
            self.vocab = Vocab(self.lines)
        else:
            self.vocab = vocab

    def __len__(self):
        return self.pos_len + self.neg_len

    def __getitem__(self, idx):
        text = self.lines[idx]
        indices = self.vocab.encode(text)
        indices = indices[:128]
        label = 1.0 if idx < self.pos_len else 0.0
        return torch.tensor(indices), torch.tensor(label)


def collate_fn(batch): # padding the sentences to align(for batch)
    texts, labels = zip(*batch) # translate the texts and labels into their kind of tuple
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0) # padding in the tensors' tuple(0 is the index of <pad>)
    return texts_padded, torch.stack(labels)

# loading weights of pre-trained glove into model
def load_glove_weights(vocab, glove_path, embedding_dim=100):
    w_matrix = np.random.normal(scale=0.6, size=(len(vocab.itos), embedding_dim))
    w_matrix[vocab.pad_idx] = np.zeros(embedding_dim)
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab.itos:
                    w_matrix[vocab.stoi[word]] = np.array(values[1:], dtype='float32')
        print('====== Successfully loaded glove weights ======\n')
    except FileNotFoundError:
        print('Warning: can not find Glove file, using random weights')
    return torch.from_numpy(w_matrix)


def train(model, device, train_loader):
    print('\n====== training ======')
    writer = SummaryWriter("2Dlogs")
    learning_rate = 0.0003
    epochs = 50

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    citerion = nn.BCEWithLogitsLoss()

    train_steps = 0
    start = time.time()
    for epoch in range(epochs):
        for input, y in train_loader:
            input = input.to(device)
            y = y.to(device)

            # with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16):
            y_hat = model(input)
            loss = citerion(y_hat.squeeze(), y.float())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_steps += 1
            if train_steps % 500 == 0:
                end = time.time()
                print(f'train_steps: {train_steps}, loss: {loss.item():.4f}, time: {(end - start):.2f}s')
                writer.add_scalar('loss', loss.item(), train_steps)

    writer.close()


def test(model, device, test_loader):
    print('\n====== testing ======')
    test_steps = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for input, y in test_loader:
            input = input.to(device)
            y = y.to(device)

            y_hat = torch.sigmoid(model(input).squeeze()).detach().cpu()
            preds = torch.round(y_hat)
            y = y.detach().cpu()
            correct += (preds == y).sum().item()

            test_steps += 1
    print(f'test_steps: {test_steps}, correct: {correct}')
    print(f'Accuracy: {correct / (test_steps * test_loader.batch_size) * 100:.2f}%')


def inference(model, device, vocab):
    print('\n====== inference ======')
    texts = ['What a terrible movie!',
             'It is worthy of its fame!',
             'I will watch it for more times!',
             'A waste of time!',
             'To be honest, I may not watch it anymore. Nevertheless, it is still worthy for watching in such a warm festival',
             'Good actors and actresses! Good skills and technologies! But I will not watch it anymore, for its terrible story...']

    for text in texts:
        tokens = vocab.encode(text)
        tokens = tokens[:64]
        tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]

        with torch.no_grad():
            output = model(tensor)
            prediction = torch.round(torch.sigmoid(output))
            print("Positive" if prediction >= 0.5 else "Negative")


def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    train_dataset = IMDBDataset(is_train=True)
    # vocab = train_dataset.vocab
    test_dataset = IMDBDataset(is_train=False, vocab=vocab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    print('\n====== vocab test ======')
    print(f'{vocab.encode("is a")}')

    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    model = BiLSTM_Sentiment(
        vocab_size=len(vocab.itos),
        embedding_dim=100,
        hidden_size=32,
        output_dim=1,
        num_layers=1,
        dropout=0.5,
    )
    glove_w = load_glove_weights(vocab, "glove.6B.100d.txt")
    model.embedding.weight.data.copy_(glove_w)
    model.to(device)

    train(model, device, train_loader)
    test(model, device, test_loader)

    print('\n====== saving model ======')
    torch.save(model.state_dict(), './GRU_2D_model.pth')
    # inference(model, device, vocab)
    # with open('vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    inference(model, device, vocab)

if __name__ == '__main__':
    main()
