src_keys = ['strength', 'length', 'phrase']
tgt_keys = ['bar', 'pos', 'token', 'dur', 'phrase']

binary_dir = './binary'
words_dir = './binary/words'

hparams = {
    'batch_size': 10,
    'word_data_dir': './binary/words',
    'sentence_maxlen': 512,
    'hidden_size': 768,
}
