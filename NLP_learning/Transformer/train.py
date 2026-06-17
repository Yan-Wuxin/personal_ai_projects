import time
import torch
import intel_extension_for_pytorch as ipex
from data_load import *
from constant import *
from model import Transformer
from torch import nn

def main():
    print('\n====== Training ======')
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    cn2idx, idx2cn = load_vocab('cn')
    en2idx, idx2en = load_vocab('en')
    X, Y = load_data('train')

    model = Transformer(len(cn2idx), len(en2idx), PAD_ID, d_model,
                        d_ff, layers, heads, dropout_rate, max_len)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID) # ignore the <PAD>
    start = time.time()
    total_steps = 0
    for epoch in range(epochs):
        for index, _ in get_batch_indices(len(X), batch_size):
            x_batch = torch.LongTensor(X[index]).to(device)
            y_batch = torch.LongTensor(Y[index]).to(device)
            y_input = y_batch[:, :-1] # offer the first n-1 words
            y_label = y_batch[:, 1:] # use 2-n words as monitor label
            y_hat = model(x_batch, y_input) # [batch, seq_len, vocab_size]

            y_label_mask = y_label != PAD_ID # [<PAD>, num, ...] -> [0, 1, ...], cleaning the effects of <PAD>
            preds = torch.argmax(y_hat, dim=-1)
            correct = preds == y_label
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = criterion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_steps += 1
            if total_steps % 100 == 0:
                end = time.time()
                print(f'Epoch:{epoch+1} Time:{end - start} Loss:{loss.item():.4f} Accuracy:{acc.item():.4f}')

    print('\n====== Saving model ======')
    model_path = './model.pth'
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()

