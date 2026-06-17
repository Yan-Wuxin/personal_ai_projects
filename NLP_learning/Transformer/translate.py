import torch
from data_load import *
from model import Transformer
import jieba
from constant import *

def main():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    cn2idx, idx2cn = load_vocab('cn')
    en2idx, idx2en = load_vocab('en')

    model = Transformer(len(cn2idx), len(en2idx), 0, d_model,
                        d_ff, layers, heads, dropout_rate, max_len)
    model.to(device)
    model.eval()
    model_path = './model.pth'
    model.load_state_dict(torch.load(model_path))

    input = '这部电影真是赏心悦目'
    x_batch = torch.LongTensor([cn2idx[x] for x in jieba.cut(input)]).to(device)

    y_input = torch.ones(batch_size, max_len, dtype=torch.long).to(device) * PAD_ID
    y_input[0] = en2idx['<S>'] # store the output

    with torch.no_grad():
        for i in range(1, y_input.shape[1]):
            y_hat = model(x_batch, y_input)
            for j in range(batch_size):
                y_input[j, i] = torch.argmax(y_hat[j, i - 1])
    output_sentence = idx2sentence(y_input[0], idx2en, True)
    print(output_sentence)

if __name__ == '__main__':
    main()

