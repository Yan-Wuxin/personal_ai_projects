import torch
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=100)

tensor = glove.get_vecs_by_tokens(['', '2026', ',', 'tensor', '20260127'], True)
print(tensor)

vocab = glove.itos
print(len(vocab))
for i in range(3):
    print(vocab[i])

def get_counterpart(x1, y1, x2):
    a1 = glove.stoi(x1)
    c1 = glove.stoi(y1)
    a2 = glove.stoi(x2)
    x1, y1, x2 = glove.get_vecs_by_tokens([x1, y1, x2], True)
    target = y1 - x1 + x2
    max_sim = 0
    max_id = -1
    for i in range(len(vocab)):
        vector = glove.get_vecs_by_tokens([vocab[i]], True)[0]
        cossim = torch.dot(target, vector)
        if cossim > max_sim and i not in [a1, c1, a2]:
            max_sim = cossim
            max_id = i
    return vocab[max_id]
print(get_counterpart('man', 'woman', 'king'))
print(get_counterpart('black', 'white', 'good'))