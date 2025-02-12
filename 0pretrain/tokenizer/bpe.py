import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os
import os.path as osp
import json
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
import time
import json
import gc
from datetime import datetime
import pdb

dnabert2_f = open('/root/pretrained/DNABERT-2-117M/tokenizer.json', 'r')
dnabert2_data = json.load(dnabert2_f)
dnabert2_vocab = dnabert2_data['model']['vocab']
dnabert2_merges = dnabert2_data['model']['merges']
dnabert2_merges = [tuple(s.split(' ')) for s in dnabert2_merges]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_dir', type=str, default='/root/projects/DNABERT_Promotor/0pretrain/tokenizer/', help='')
    parser.add_argument('--data_dir', type=str, default='/hy-tmp/prokaryotes/bpe_train', help='')
    parser.add_argument('--vocab_size', type=int, default=4096, help='')
    parser.add_argument('--num_workers', type=int, default=5, help='default:4')  # 5
    parser.add_argument('--batch_size', type=int, default=1000, help='')  # 1000
    
    args = parser.parse_args()
    return args

def get_current_time():
    current_time = str(datetime.fromtimestamp(int(time.time())))
    d = current_time.split(' ')[0]
    t = current_time.split(' ')[1]
    d = d.split('-')[0] + d.split('-')[1] + d.split('-')[2]
    t = t.split(':')[0] + t.split(':')[1] + t.split(':')[2]
    return d+ '_' +t

def train_part(data_part, trainer):
    # bpe = BPE(unk_token="[UNK]")
    bpe = BPE(dnabert2_vocab, dnabert2_merges)
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(data_part, trainer)
    return tokenizer

def main_multiprocess(args):
    # data
    filenamelist = os.listdir(args.data_dir)
    paths = [osp.join(args.data_dir, filename) for filename in filenamelist]
    # paths = paths[:50]
    data_parts = [paths[i:i + args.batch_size] for i in range(0, len(paths), args.batch_size)]

    # dir
    current_time = get_current_time()
    tokenizer_name = current_time + '_tokenizer' + str(args.vocab_size) + '_multiprocess'
    tokenizer_dir = osp.join(args.tokenizer_dir, tokenizer_name)
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
        vocab_size = args.vocab_size, 
        min_frequency=2,
        show_progress=True
    )
    
    # train part data by part_tokenizers
    start = time.time()
    for i in range(0, len(data_parts), args.num_workers):
        if (i+args.num_workers) < len(data_parts):
            data_parts_i = data_parts[i:i+args.num_workers]
        else:
            data_parts_i = data_parts[i:]

        with Pool(args.num_workers) as pool:
            tokenizers = pool.starmap(train_part, [(part, trainer) for part in data_parts_i])
        
        for j in range(len(tokenizers)):
            tokenizers[j].save(os.path.join(tokenizer_dir, "tokenizer_part" + str(i) + str(j) + ".json"))
        gc.collect()
    elapse = time.time()-start
    print('Time: {}'.format(round(elapse, 3)))
    print("Part training finished.")
    print()
    print()
    
    # get vocabs & merges from part_tokenizers
    final_vocab = Counter()
    final_merges = Counter()
    tokenizers_part_filenames = os.listdir(tokenizer_dir)
    for filename in tokenizers_part_filenames:
        f = open(osp.join(tokenizer_dir, filename), 'r')
        data = json.load(f)
        vocab = data['model']['vocab']
        merges = data['model']['merges']
        
        final_vocab.update(vocab)
        for merge in merges:
            final_merges[merge] += 1
    final_vocab = dict(final_vocab)
    final_merges = [tuple(s.split(' ')) for s in list(final_merges.keys())]

    # train final tokenizer by total data
    final_tokenizer = Tokenizer(BPE(final_vocab, final_merges))
    final_tokenizer.pre_tokenizer = Whitespace()
    
    for i in range(0, len(data_parts)):
        final_tokenizer.train(data_parts[i], trainer)
        gc.collect()
    
    final_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", final_tokenizer.token_to_id("[CLS]")),
            ("[SEP]", final_tokenizer.token_to_id("[SEP]")),
        ],
    )
    final_tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    elapse = time.time()-elapse
    print('Time: {}'.format(round(elapse, 3)))
    print("Merge training finished.")
        

    # generate and save tokenizer config
    tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast", 
                        "unk_token": "[UNK]",
                        "cls_token": "[CLS]",
                        "sep_token": "[SEP]",
                        "pad_token": "[PAD]",
                        "mask_token": "[MASK]"}
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)
    
    print("Tokenizer saved.")

def merge():
    root = '/root/projects/DNABERT_Promotor/0pretrain/tokenizer/'
    new_tokenizer_dir = osp.join(root, '20240603_093134_tokenizer4096_multiprocess/tokenizer.json')
    dnabert2_tokenizer_dir = '/root/pretrained/DNABERT-2-117M/tokenizer.json'
    final_vocab = Counter()
    final_merges = Counter()
    tokenizers_part_filenames = [new_tokenizer_dir, dnabert2_tokenizer_dir]
    for filename in tokenizers_part_filenames:
        f = open(filename, 'r')
        data = json.load(f)
        vocab = data['model']['vocab']
        merges = data['model']['merges']
        
        final_vocab.update(vocab)
        for merge in merges:
            final_merges[merge] += 1
    sorted_vocab_items = sorted(final_vocab.items(), key=lambda item: item[1])
    final_vocab_ordered = {item[0]: idx for idx, item in enumerate(sorted_vocab_items)}
    
    final_merges = [tuple(s.split(' ')) for s in list(final_merges.keys())]

    current_time = get_current_time()
    tokenizer_name = current_time + '_tokenizer' + str(len(final_vocab_ordered)) + '_merge_dnabert2'
    tokenizer_dir = osp.join(root, tokenizer_name)
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)

    final_tokenizer = Tokenizer(BPE(final_vocab_ordered, final_merges))
    final_tokenizer.added_tokens = data['added_tokens']
    final_tokenizer.pre_tokenizer = Whitespace()
    final_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", final_tokenizer.token_to_id("[CLS]")),
            ("[SEP]", final_tokenizer.token_to_id("[SEP]")),
        ],
    )
    final_tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    # generate and save tokenizer config
    tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast", 
                        "unk_token": "[UNK]",
                        "cls_token": "[CLS]",
                        "sep_token": "[SEP]",
                        "pad_token": "[PAD]",
                        "mask_token": "[MASK]"}
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)
        
    # generate and save model config
    with open('/root/pretrained/DNABERT-2-117M/config.json', "r") as f:
        model_config = json.load(f)
    model_config['vocab_size'] = len(final_vocab_ordered)
    with open(os.path.join(tokenizer_dir, "config.json"), "w") as f:
        json.dump(model_config, f)

def get_genes_txts():
    root = '/hy-tmp/prokaryotes/'
    
    genes_path = osp.join(root, 'genes/')
    bpe_train_path = osp.join(root, 'bpe_train/')
    if not os.path.exists(bpe_train_path):
        os.makedirs(bpe_train_path)
    
    filenamelist = os.listdir(genes_path)
    

    for i in range(len(filenamelist)):
        seqs = []
        filepath = osp.join(genes_path, filenamelist[i])
        lines = open(filepath, 'r')
        for line in lines:
            line = line.replace('\n', '').split(',')
            
            promoter = line[1]
            gene = line[2][:200]
            seq = promoter + gene
            seqs.append(seq)
        # pdb.set_trace()
        np.savetxt(osp.join(bpe_train_path, filenamelist[i]), seqs, fmt='%s')
        print(i+1, filenamelist[i], 'done.')

def get_genes_txt():
    root = '/hy-tmp/prokaryotes/'
    
    genes_path = osp.join(root, 'genes/')
    bpe_train_path = osp.join(root, 'bpe_train.txt')
    f = open(bpe_train_path, 'a+')
    
    # if not os.path.exists(bpe_train_path):
    #     os.makedirs(bpe_train_path)
    
    filenamelist = os.listdir(genes_path)

    for i in range(len(filenamelist)):
        filepath = osp.join(genes_path, filenamelist[i])
        lines = open(filepath, 'r')
        for line in lines:
            line = line.replace('\n', '').split(',')
            
            promoter = line[1]
            gene = line[2][:200]
            seq = promoter + gene
            f.write(seq)
            # seqs.append(seq)
        # pdb.set_trace()
        print(i+1, filenamelist[i], 'done.')
    # np.savetxt(osp.join(bpe_train_path, filenamelist[i]), seqs, fmt='%s')
    f.close()
            
if __name__ == '__main__':
    # get_genes_txts()
    # get_genes_txt()

    args = parse()
    # main_multiprocess(args)
    merge()