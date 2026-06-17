import os
import re

def read_imdb_vocab(dir='LetterRNN/data/aclImdb'):
    fn = os.path.join(dir, 'imdb.vocab')
    with open(fn, 'rb') as f: # use regular expression to obtain words only containing letter and ' '
        word = f.read().decode('utf-8').replace('\n', ' ')
        words = re.sub(u'([^\u0020\u0061-\u007a])', '', word.lower()).split(' ')
        words = [w for w in words if len(w) > 0]
    return words

