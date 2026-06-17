import intel_extension_for_pytorch as ipex
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import RNN
from constant import *
from read_imdb import read_imdb_vocab

class WordDataset(Dataset):
    def __init__(self, words, max_len):
        super().__init__()
        self.words = words
        self.max_len = max_len
        # self.is_onedot = is_onedot

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx] + " "
        length = len(word)
        # if self.is_onedot: # using one-dot encoding
        #     tensor = torch.zeros(self.max_len, EMBEDDING_LENGTH)
        #     for i in range(self.max_len):
        #         if i < length:
        #             tensor[i][LETTER_MAP[word[i]]] = 1
        #         else:
        #             tensor[i][0] = 1 # regard the remaining space as ' '
        # else: # using normal encoding
        tensor = torch.zeros(self.max_len, dtype=torch.long)
        for i in range(length):
            tensor[i] = LETTER_MAP[word[i]]
        return tensor



def words_to_labels(words, max_len):
    if isinstance(words, str):
        words = [words]
    words = [w + ' ' for w in words]
    size = len(words)
    tensor = torch.zeros(size, max_len, dtype=torch.long)
    for i in range(size):
        for j, letter in enumerate(words[i]):
            tensor[i][j] = LETTER_MAP[letter]
    return tensor



def get_dataloader_and_max_len(limit_length=None):
    words = read_imdb_vocab()
    max_len = 0
    for word in words:
        max_len = max(max_len, len(word))
    if limit_length is not None and max_len > limit_length:
        words = [w for w in words if len(w) <= limit_length]
        max_len = limit_length
    max_len += 1 # <EOS>
    dataset = WordDataset(words, max_len)
    print(f'\ntraining dataset size:{len(dataset)} max word len:{max_len}')
    return DataLoader(dataset, batch_size=256), max_len



def train(device, model):
    print(f'\n====== training ======')
    writer = SummaryWriter("logs")
    learning_rate = 1e-3
    loader, max_len = get_dataloader_and_max_len()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    citerion = nn.CrossEntropyLoss()

    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    epochs = 5
    train_steps = 0
    start_time = time.time()
    for epoch in range(epochs):
        for y in loader:
            train_steps += 1
            y = y.to(device)
            y_hat = model(y)

            batch, Tx, _ = y_hat.shape
            y_hat = torch.reshape(y_hat, (batch * Tx, -1))
            y_label = torch.reshape(y, (batch * Tx, ))
            loss = citerion(y_hat, y_label) # can not switch two characters

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) ###
            optimizer.step()

            if train_steps % 100 == 0:
                end = time.time()
                print(f'\ntraining step: {train_steps}, loss: {loss}, time: {end - start_time}')
                writer.add_scalar('train_loss', loss.item(), train_steps)

    writer.close()



def test(model, device):
    print(f'\n====== testing ======')
    model.eval()
    test_data = [
        'apple', 'appll', 'appla', 'apply', 'bear', 'beer', 'berr', 'beee', 'car',
        'cae', 'cat', 'cac', 'caq', 'query', 'queee', 'queue', 'queen', 'quest',
        'quess', 'quees'
    ]
    print(f'\n{len(test_data)} test samples')
    _, max_len = get_dataloader_and_max_len()
    test_words = words_to_labels(test_data, max_len)
    test_words = test_words.to(device)
    probs = model.language_model(test_words)
    for word, prob in zip(test_data, probs):
        print(f'\n{word}: {prob}')



def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f'gpu availability: {torch.xpu.is_available()}\n')
    model = RNN().to(device)
    print('\n====== model ======\n')
    print(model)

    train(device, model)
    test(model, device)

    print('\n====== saving ======')
    torch.save(model.state_dict(), 'RNN_model.pth')

if __name__ == '__main__':
    main()