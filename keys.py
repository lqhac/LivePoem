src_keys = ['strength', 'length', 'phrase']
tgt_keys = ['bar', 'pos', 'token', 'dur', 'phrase']

binary_dir = '/home/qihao/CS6207/binary'
words_dir = '/home/qihao/CS6207/binary/words'

hparams = {
    'batch_size': 10,
    'word_data_dir': '/home/qihao/CS6207/binary/words',
    'sentence_maxlen': 512,
    'hidden_size': 768,
}