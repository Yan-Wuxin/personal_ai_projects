import codecs
import os

import numpy as np
import requests
from jupyter_core.migrate import regex
from torch import random

min_cnt = 0 # the baseline of <UNK>
max_len = 50 # maximum number in a sentence
data_dir = 'Transformer/data'


def load_vocab(language):
    assert language in ['cn', 'en']
    vocab = [
        line.split()[0] for line in codecs.open(
            os.path.join(data_dir, f'{language}.txt'),
            'r',
            'utf-8'
        ).read().splitlines()
        if int(line.split()[1]) >= min_cnt
    ]
    word2idx = {word : i for i, word in enumerate(vocab)}
    idx2word = [word for word in vocab]
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    cn2idx, idx2cn = load_vocab('cn')
    en2idx, idx2en = load_vocab('en')
    # idx
    x_list, y_list, sources, targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [
            en2idx.get(w, 1)
            for w in ('<S>' + source_sent + '</S>').split()
        ]
        y = [
            cn2idx.get(w, 1)
            for w in ('<S>' + target_sent + '</S>').split()
        ]
        if max(len(x), len(y)) < max_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            sources.append(source_sent)
            targets.append(target_sent)
    # pad
    X = np.zeros([len(x_list), max_len], np.int32)
    Y = np.zeros([len(y_list), max_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, max_len - len(x)], 'constant', constant_values=(0,0))
        Y[i] = np.lib.pad(x, [0, max_len - len(y)], 'constant', constant_values=(0,0))
    return X, Y, sources, targets


def load_data(type):
    assert type in ['train', 'test']
    if type == 'train':
        source, target = os.path.join(data_dir, 'cn.txt'), os.path.join(data_dir, 'en.txt')
    else:
        source, target = os.path.join(data_dir, 'cn.test.txt'), os.path.join(data_dir, 'en.test.txt')
    cn_sents = [
        regex.sub("[^\s\p{L}']", '', line)
        for line in codecs.open(source, 'r', 'utf-8').read().splitlines()
        if line and line[0] != '<'
    ]
    en_sents = [
        regex.sub("[^\s\p{L}']", '', line)
        for line in codecs.open(target, 'r', 'utf-8').read().splitlines()
        if line and line[0] != '<'
    ]
    X, Y, source, target = create_data(cn_sents, en_sents)
    return X, Y


def get_batch_indices(total_length, batch_size):
    assert (batch_size <= total_length), ('Batch size is larger than data length.')
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index:current_index + batch_size], current_index


def idx2sentence(arr, vocab, insert_space=False):
    res = ''
    first_word = True
    for id in arr:
        word = vocab[id.item()]
        if insert_space and not first_word: # insert space between words
            res += ' '
        first_word = False
        res += word
    return res


def download(url, dir):
    os.makedirs(dir, exist_ok=True)
    name = url.split('/')[-1]
    path = os.path.join(dir, name)
    if not os.path.exists(path):
        print(f'Installing {name}...')
        open(path, 'wb').write(requests.get(url).content) ##
        print('Install successfully')


def download_data():
    urls = [('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/corpora/cn.txt'),
            ('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/corpora/en.txt'),
            ('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/preprocessed/cn.txt.vocab.tsv'),
            ('https://raw.githubusercontent.com/P3n9W31/transformer-pytorch/'
             'master/preprocessed/en.txt.vocab.tsv')]
    for url in urls:
        download(url, data_dir)

if __name__ == '__main__':
    download_data()