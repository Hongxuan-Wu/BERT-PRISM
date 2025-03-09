import os
import os.path as osp
import time
import transformers
import numpy as np
import pandas as pd
import pdb


def getTokens():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    # model_name_or_path = '/root/pretrained/DNABERT-2-117M'
    model_name_or_path = '/root/pretrained/checkpoint-720000/'
    
    model_max_length = 100
    data_path = '/hy-tmp/prokaryotes/genes/'
    # token_path = '/hy-tmp/prokaryotes/eco_tokens_dnabert2/'
    token_path = '/hy-tmp/promoter_real_fake_predict/genes_ecos/'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    start = time.time()
    
    f = open('/root/projects/DNABERT_Promotor/0pretrain/preprocess/genomes_filtered.csv')
    file = f.readlines()
    eco_names = [line.split(',')[1] for line in file][0:68]
    f.close()
    filenamelist = [name for name in os.listdir(data_path) if name.split('.')[0] in eco_names]

    if not os.path.exists(token_path):
        os.mkdir(token_path)
    
    for i in range(len(filenamelist)):
    # for i in range(0, 68):  # ecos
        gene_path = osp.join(data_path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()
    
        sequences = []
        for line in file:
            line = line.replace('\n', '').split(',')
            # seq = line[1] + line[2]
            seq = line[1] + gene.upper()
            sequences.append(seq)
        output = tokenizer(
            sequences, 
            return_tensors="np", 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            truncation=True
        )
        data = output['input_ids']
        np.save(os.path.join(token_path, filenamelist[i].split('.')[0]+'.npy'), data)
        print(str(i), 'saved.')
        # pdb.set_trace()
    print("Token saving time: ", time.time()-start, 's.')

def getTokensFromCSV():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    # model_name_or_path = '/root/pretrained/DNABERT-2-117M'
    model_name_or_path = '/root/pretrained/checkpoint-720000/'
    
    model_max_length = 100
    data_path = '/hy-tmp/promoter_real_fake_predict/'
    # token_path = '/hy-tmp/prokaryotes/eco_tokens_dnabert2/'
    # token_path = '/hy-tmp/promoter_real_fake_predict/genes_ecos/'
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    start = time.time()
    
    f = open(osp.join(data_path, 'genes_ecos.csv'))
    lines = f.readlines()[1:]
    f.close()
    
    sequences = [line.replace('\n', '').split(',')[0] for line in lines]
    labels = [int(line.replace('\n', '').split(',')[1]) for line in lines]
    
    output = tokenizer(
        sequences, 
        return_tensors="np", 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True
    )
    
    data = output['input_ids']
    masks = output['attention_mask']
    # np.save(os.path.join(data_path, 'genes_ecos.npy'), data)
    np.savez(os.path.join(data_path, 'genes_ecos.npz'), **{'sequences':data, 'masks': masks, 'labels': labels})
    # np.savez_compressed(os.path.join(data_path, 'genes_ecos_compressed.npz'), **{'sequences':data, 'masks': masks, 'labels': labels})
    print('saved.')
    # pdb.set_trace()
    print("Token saving time: ", time.time()-start, 's.')
    
def checkTokens():
    data_path = '/hy-tmp/prokaryotes/genes/'
    
    token_path1 = '/hy-tmp/prokaryotes/eco_tokens/'
    file1 = np.load(osp.join(token_path1, 'T00007.npy'))

    token_path2 = '/hy-tmp/prokaryotes/eco_tokens_dnabert2/'
    file2 = np.load(osp.join(token_path2, 'T00007.npy'))
    
    token_path3 = '/hy-tmp/prokaryotes/tokens/'
    file3 = np.load(osp.join(token_path3, 'T00007.npy'))
    
    pdb.set_trace()
    
def add5UTR():
    root = '/root/projects/DNABERT_Promotor/1train/promoter_real_fake_predict/data/csvs/'
    
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    UTR_5 = 'TCCGGATTTACTAACTGGAAGGGGGACTAA'
    
    path = 'csv_cat/Escherichia coli str K-12 substr. MG1655 promoters.csv'
    df = pd.read_csv(osp.join(root, path), delimiter=',')
    sequences = []
    
    for i in range(len(df)):
        sequences.append(df['PromoterSeq'][i][:81] + UTR_5 + gene.upper())
    df['PromoterSeq'] = sequences
    
    path2 = 'csv_cat2/Escherichia coli str K-12 substr. MG1655 promoters.csv'
    if not os.path.exists(os.path.join(root, 'csv_cat2/')):
        os.makedirs(os.path.join(root, 'csv_cat2/'))
    
    df.to_csv(os.path.join(root, path2), sep=',', index=False, header=True)
    
    # pdb.set_trace()


if __name__ == "__main__":
    # getTokens()
    # checkTokens()
    # getTokensFromCSV()
    # add5UTR()
    pass