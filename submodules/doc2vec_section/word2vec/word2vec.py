#!/usr/bin/python
import argparse
import itertools
from pickle import TRUE

from word2vec.data import batch_cbow, batch_cbsec, doc
from word2vec.model import cbow, cbsec, model
from word2vec import vocab

import json
import pandas


MODEL_TYPES = {
    'cbow': (cbow.CBOW, batch_cbow.data_generator, batch_cbow.batch),
    'cbsec': (cbsec.CBSEC, batch_cbsec.data_generator, batch_cbsec.batch)
}


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='Path to documents directory')

    parser.add_argument('--model', default='cbow',
                        choices=list(MODEL_TYPES.keys()),
                        help='Which model to use')
                        
    parser.add_argument('--save', help='Path to save model')
    parser.add_argument('--save_period', 
                        type=int,
                        help='Save model every n epochs')
    parser.add_argument('--save_vocab', help='Path to save vocab file')
    parser.add_argument('--save_word_embeddings',
                        help='Path to save doc embeddings file')
    parser.add_argument('--save_word_embeddings_period',
                        type=int,
                        help='Save doc embeddings every n epochs')

    parser.add_argument('--load', help='Path to load model')
    parser.add_argument('--load_vocab', help='Path to load vocab file')

    parser.add_argument('--early_stopping_patience',
                        type=int,
                        help='Stop after no loss decrease for n epochs')

    parser.add_argument('--vocab_size', default=vocab.DEFAULT_SIZE,
                        type=int,
                        help='Max vocabulary size; ignored if loading from file')
    parser.add_argument('--vocab_rare_threshold',
                        default=vocab.DEFAULT_RARE_THRESHOLD,
                        type=int,
                        help=('Words less frequent than this threshold '
                              'will be considered unknown'))

    parser.add_argument('--window_size',
                        default=model.DEFAULT_WINDOW_SIZE,
                        type=int,
                        help='Context window size')
    parser.add_argument('--embedding_size',
                        default=model.DEFAULT_EMBEDDING_SIZE,
                        type=int,
                        help='Word and document embedding size')

    parser.add_argument('--num_epochs',
                        default=model.DEFAULT_NUM_EPOCHS,
                        type=int,
                        help='Number of epochs to train for')
    parser.add_argument('--steps_per_epoch',
                        default=model.DEFAULT_STEPS_PER_EPOCH,
                        type=int,
                        help='Number of samples per epoch')
    parser.add_argument('--json_num',
                        default=1,
                        type=int)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', dest='train', action='store_true')
    group.add_argument('--no-train', dest='train', action='store_false')
    group.set_defaults(train=False)

    return parser.parse_args()

def sections_by_id(path, num):
    datas = []
    for i in range(num):
        with open(path + "/koreaherald_1517_" + str(i) + ".json", 'r') as f:
            datas += [pandas.DataFrame.from_dict(json.load(f))]

    df = pandas.concat(datas)
    sections = {}
    sec = ['North Korea', 'Social affairs', 'Defense', 'Foreign Policy', 'Diplomatic Circuit', 'Politics', 'Foreign  Affairs', 'National', 'Science', 'Education', 'International']
    for i in range(len(df.index)):
        sections[i] = 11
        for j in range(len(sec)):
            if sec[j] in df[' section'][i]:
                sections[i] = j
    return sections

def main():
    args = _parse_args()
    if args.train:
        tokens_by_doc_id = doc.tokens_by_doc_id(args.path, args.json_num, isbody=True)
    else:
        tokens_by_doc_id = doc.tokens_by_doc_id(args.path, args.json_num, isbody=False)

    num_docs = len(tokens_by_doc_id)

    v = vocab.Vocabulary()
    if args.load_vocab:
        v.load(args.load_vocab)
    else:
        all_tokens = list(itertools.chain.from_iterable(tokens_by_doc_id.values()))
        v.build(all_tokens, max_size=args.vocab_size)
        if args.save_vocab:
            v.save(args.save_vocab)

    token_ids_by_doc_id = {d: v.to_ids(t) for d, t in tokens_by_doc_id.items()}

    model_class, data_generator, batcher = MODEL_TYPES[args.model]

    m = model_class(args.window_size, v.size, num_docs,
                    embedding_size=args.embedding_size)

    if args.load:
        m.load(args.load) 
    else:
        m.build()
        m.compile()

    elapsed_epochs = 0

    if args.train:
        if args.model == "cbsec": 
            section_ids_by_doc_id = sections_by_id(args.path, args.json_num)
            all_data = batcher(
                    data_generator(
                        token_ids_by_doc_id,
                        args.window_size,
                        v.size, section_ids_by_doc_id))
        else:
            all_data = batcher(
                    data_generator(
                        token_ids_by_doc_id,
                        args.window_size,
                        v.size))

        history = m.train(
                all_data,
                epochs=args.num_epochs,
                steps_per_epoch=args.steps_per_epoch,
                early_stopping_patience=args.early_stopping_patience,
                save_path=args.save,
                save_period=args.save_period,
                save_word_embeddings_path=args.save_word_embeddings,
                save_word_embeddings_period=args.save_word_embeddings_period)

        elapsed_epochs = len(history.history['loss'])

    if args.save:
        m.save(
            args.save.format(epoch=elapsed_epochs))

    if args.save_word_embeddings:
        m.save_word_embeddings(
            args.save_word_embeddings.format(epoch=elapsed_epochs))
