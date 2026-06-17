import os

def read_imdb(dir='../data/aclImdb', split='pos', is_train=True):
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    lines = []
    for file in os.listdir(dir): #
        with open(os.path.join(dir, file), 'r', encoding='utf-8') as f:
            line = f.readline()
            lines.append(line)
    # print(lines[0])
    return lines
