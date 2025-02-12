import os
import os.path as osp
import time
import numpy as np
import pandas as pd
import transformers
import pdb
from Bio import SeqIO
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

root = '/data/whx/projects/DNABERT_Promotor/1train/expression_intensity_classification/data/ecos/'
    
def pre_ecos_cat():
    df1 = pd.read_csv('/hy-tmp/ecos/expression_fromlist.csv')
    df2 = pd.read_csv('/hy-tmp/ecos/expression_manual.csv')
    df = pd.concat([df1, df2]).reset_index(drop=True)
    df.columns = ['geneId', 'promoter', 'gene', 'tpm']
    
    # pdb.set_trace()
    # gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    # df['genes'] = gene.upper()
    
    df.to_csv('/hy-tmp/ecos/expression_ecos.csv', sep=',', index=False, header=True)
    
def getTokensFromCSV():
    # model_name_or_path = '/data/whx/ckpt/DNABERT-2-117M'
    model_name_or_path = '/data/whx/ckpt/checkpoint-720000/'
    # model_name_or_path = '/data/whx/ckpt/checkpoint-720000_newtoken/'
    # model_name_or_path = '/data/whx/ckpt/checkpoint-720000_mergetoken/'

    dataset = 'genes_prokaryotes'
    # dataset = 'genes_escherichia'
    # dataset = 'genes_streptomyces'
    # dataset = 'genes_vibrio'

    catATG = True
    # catATG = False

    root_path = '/data/whx/expressions/data_gene_tpm/'


    if 'prokaryotes' in dataset:
        dataname = 'tpm_gene_' + dataset[6:] + '_filtered'
    else:            
        dataname = 'tpm_gene_' + dataset[6:]
    
    read_path = osp.join(root_path, dataname+'.txt')
    
    if 'newtoken' in model_name_or_path:
        dataname += '_newtoken'
    elif 'mergetoken' in model_name_or_path:
        dataname += '_mergetoken'
    else:
        pass
    
    if catATG and ('prokaryotes' not in dataset):
        dataname += '_catATG'
    
    save_path = osp.join(root_path, dataname+'.npz')
    
    # pdb.set_trace()
    
    df = pd.read_csv(read_path)
    if catATG:
        gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
        df['gene'] = gene.upper()
    sequences = (df['promoter'] + df['gene']).tolist()
    intensity = (df['tpm']).tolist()
    
    
    start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=100,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    output = tokenizer(
        sequences, 
        return_tensors="np", 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True
    )
    data = output['input_ids']
    masks = output['attention_mask']
    np.savez_compressed(save_path, **{'sequences':data, 'masks': masks, 'intensity': intensity})
    
    print('saved.')
    print("Token saving time: ", time.time()-start, 's.')
    

def pre_MG1655_RegulonDB_strong_weak():
    f = open(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB.csv'))
    file = f.readlines()[28:]
    f.close()
    # head = file[0].replace(' \n', '').split(',')
    lines = file[1:]
    
    promoters_real = []
    for line in lines:
        line_cut = line.split(',')
        promoter = line_cut[5]
        if promoter == 'None':
            continue
        
        if line_cut[10] == 'S':
            promoters_real.append([promoter, 1])
        elif line_cut[10] == 'W':
            promoters_real.append([promoter, 0])
        else:
            continue
    promoters_real = np.array(promoters_real)
    # pdb.set_trace()
    
    # get fake
    # path = '/hy-tmp/genes/'
    # filenamelist = os.listdir(path)
    # genes = []
    # for i in range(len(filenamelist)):
    #     gene_path = osp.join(path, filenamelist[i])
    #     f = open(gene_path)
    #     file = f.readlines()
    #     f.close()
    #     for line in file:
    #         line = line.replace('\n', '').split(',')
    #         genes.append(line[2])
    # # pdb.set_trace()
    
    # cut_length = len(promoters_real[0][0])
    # cut_genes = []
    # for i in range(len(promoters_real)):
    #     chosenID = random.randint(0, len(genes)-1)
    #     while (len(genes[chosenID])-200) < cut_length:
    #         chosenID = random.randint(0, len(genes)-1)
    #     start = random.randint(200, len(genes[chosenID]) - cut_length)
    #     cut_gene = genes[chosenID][start:start+cut_length]            
    #     cut_genes.append([cut_gene, 0])
    # promoters_fake = np.array(cut_genes)
    
    # promoters = np.concatenate([promoters_real, promoters_fake], axis=0)
    # df = pd.DataFrame(promoters, columns=['sequence', 'label'])
    df = pd.DataFrame(promoters_real, columns=['promoters', 'intensity_cls'])
    
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_strong_weak.csv'), sep=',', index=False, header=True)

def pre_MG1655_RegulonDB_promoter_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    df = pd.read_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters_cat5UTRATG.csv'), sep=',', index=False, header=True)
    
    df = pd.read_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = gene.upper()
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters_catATG.csv'), sep=',', index=False, header=True)

def pre_MG1655_RegulonDB_strong_weak_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'

    df = pd.read_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_strong_weak.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_strong_weak_cat5UTRATG.csv'), sep=',', index=False, header=True)

    df = pd.read_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_strong_weak.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = gene.upper()
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_strong_weak_catATG.csv'), sep=',', index=False, header=True)

def pre_gnn210095_sup_0002_datas1():
    file = pd.read_excel(
        os.path.join(root, 'gnn210095/ggn210095-sup-0002-datas1.xlsx'), 
        sheet_name='RegulonDB',
        engine='openpyxl',
        keep_default_na=False,
        header=None
    )
    file.columns = ['promoters', 'intensity_cls']
    
    strong = file[file.loc[:, 'intensity_cls'] == 'Strong']
    strong.loc[:, 'intensity_cls'] = 1
    weak = file[file.loc[:, 'intensity_cls'] == 'Weak']
    weak.loc[:, 'intensity_cls'] = 0
    
    df = pd.concat([strong, weak])
    df.to_csv(osp.join(root, 'gnn210095/ggn210095-sup-0002-datas1_strong_weak.csv'), sep=',', index=False, header=True)
    # pdb.set_trace()
 
def pre_gnn210095_sup_0002_datas1_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    df = pd.read_csv(osp.join(root, 'gnn210095/ggn210095-sup-0002-datas1_strong_weak.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(osp.join(root, 'gnn210095/ggn210095-sup-0002-datas1_strong_weak_cat5UTRATG.csv'), sep=',', index=False, header=True)
    
    df = pd.read_csv(osp.join(root, 'gnn210095/ggn210095-sup-0002-datas1_strong_weak.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = gene.upper()
    df.to_csv(osp.join(root, 'gnn210095/ggn210095-sup-0002-datas1_strong_weak_catATG.csv'), sep=',', index=False, header=True)

def pre_mg1655_promoter():
    RegulonDB_path = osp.join(root, '1-s2.0-S088875431830613X-mmc1')
    
    with open(osp.join(RegulonDB_path, 'strong1591.txt'), 'r') as f:
        lines = f.readlines()
        strong = []
        for line in lines:
            tmp = line.replace('\n', '').replace('\t', ' ').split(' ')
            tmp = tmp[-1].replace(' ', '')
            strong.append(tmp)
    f.close()
    np.savetxt(osp.join(RegulonDB_path, 'strong.txt'), strong, fmt='%s')
    
    with open(osp.join(RegulonDB_path, 'weak1791.txt'), 'r') as f:
        lines = f.readlines()
        weak = []
        for line in lines:
            tmp = line.replace('\n', '').replace('\t', ' ').split(' ')
            tmp = tmp[-1].replace(' ', '')
            weak.append(tmp)
    f.close()
    np.savetxt(osp.join(RegulonDB_path, 'weak.txt'), weak, fmt='%s')
    
    with open(osp.join(RegulonDB_path, 'non3382.txt'), 'r') as f:
        lines = f.readlines()
        non = []
        for line in lines:
            tmp = line.replace('\n', '').replace('\t', ' ').split(' ')
            tmp = ''.join(tmp[3:]).replace(' ', '')
            non.append(tmp)
    f.close()
    np.savetxt(osp.join(RegulonDB_path, 'non.txt'), non, fmt='%s')
    
    # promoters_strong = np.array([[strong[i], 1] for i in range(len(strong))])
    # promoters_weak = np.array([[weak[i], 1] for i in range(len(weak))])
    # promoters_non = np.array([[non[i], 0] for i in range(len(non))])
    # promoters = np.concatenate([promoters_strong, promoters_weak, promoters_non], axis=0)
    # df = pd.DataFrame(promoters, columns=['promoter', 'label'])
    # df.to_csv(osp.join(RegulonDB_path, 'iPSW2L_PseKNC_mg1655.csv'), sep=',', index=False, header=True)

    promoters_strong = np.array([[strong[i], 1] for i in range(len(strong))])
    promoters_weak = np.array([[weak[i], 0] for i in range(len(weak))])
    promoters = np.concatenate([promoters_strong, promoters_weak], axis=0)
    df = pd.DataFrame(promoters, columns=['promoters', 'intensity_cls'])
    df.to_csv(osp.join(RegulonDB_path, 'iPSW2L_PseKNC_mg1655_strong_weak.csv'), sep=',', index=False, header=True)

def pre_mg1655_promoter_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    df = pd.read_csv(osp.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655.csv'))
    df['sequence'] =  df['sequence'].str.upper() + 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(os.path.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_cat5UTRATG.csv'), sep=',', index=False, header=True)

    df = pd.read_csv(osp.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655.csv'))
    df['sequence'] =  df['sequence'].str.upper() + gene.upper()
    df.to_csv(os.path.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_catATG.csv'), sep=',', index=False, header=True)

def pre_mg1655_strong_weak_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    df = pd.read_csv(osp.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_strong_weak.csv'))
    # df['promoter'] =  df['promoter'].str.upper() + 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(os.path.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_strong_weak_cat5UTRATG.csv'), sep=',', index=False, header=True)

    df = pd.read_csv(osp.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_strong_weak.csv'))
    df['promoters'] =  df['promoters'].str.upper()
    df['genes'] = gene.upper()
    df.to_csv(os.path.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_strong_weak_catATG.csv'), sep=',', index=False, header=True)


if __name__ == "__main__":
    # pre_ecos_cat()
    getTokensFromCSV()
    # pre_MG1655_RegulonDB_strong_weak()
    # pre_MG1655_RegulonDB_promoter_cat()
    # pre_MG1655_RegulonDB_strong_weak_cat()
    
    # pre_gnn210095_sup_0002_datas1()
    # pre_gnn210095_sup_0002_datas1_cat()
    
    # pre_mg1655_promoter()
    # pre_mg1655_strong_weak_cat()
    # pre_mg1655_promoter_cat()
    
    pass
