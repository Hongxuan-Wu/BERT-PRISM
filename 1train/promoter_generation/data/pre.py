import os
import os.path as osp
import numpy as np
import pandas as pd
import random
import time
import transformers
import pdb

root = '/root/projects/DNABERT_Promotor/1train/promoter_real_fake_predict/data/'

def pre_ESM_promoter():
    # get real
    file = pd.read_excel(
        os.path.join(root, '41592_2018_BFnmeth4633_MOESM3_ESM/41592_2018_BFnmeth4633_MOESM3_ESM.xlsx'), 
        sheet_name='Metadata',
        engine='openpyxl',
        keep_default_na=False
    )
    file['label'] = 1
    promoters = file[['Regulatory Sequence', 'label']].to_numpy()
    # np.savetxt(osp.join(root, '41592_2018_BFnmeth4633_MOESM3_ESM/real.txt'), promoters, fmt='%s')

    # get fake
    path = '/hy-tmp/genes/'
    filenamelist = os.listdir(path)    
    genes = []
    for i in range(len(filenamelist)):
        gene_path = osp.join(path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()
        for line in file:
            line = line.replace('\n', '').split(',')
            genes.append(line[2])

    cut_genes = []
    cut_length = len(promoters[0][0])
    for i in range(len(promoters)):
        chosenID = random.randint(0, len(genes)-1)
        while (len(genes[chosenID])-200) < cut_length:
            chosenID = random.randint(0, len(genes)-1)
        start = random.randint(200, len(genes[chosenID]) - cut_length)
        cut_gene = genes[chosenID][start:start+cut_length]
        cut_genes.append([cut_gene, 0])
    cut_genes = np.array(cut_genes)
    
    promoters = np.concatenate([promoters, cut_genes], axis=0)
    df = pd.DataFrame(promoters, columns=['sequence', 'label'])
    # np.savetxt(osp.join(root, '41592_2018_BFnmeth4633_MOESM3_ESM/fake.txt'), cut_genes, fmt='%s')
    df.to_csv(osp.join(root, '41592_2018_BFnmeth4633_MOESM3_ESM/Nathan_multi_mixed.csv'), sep=',', index=False, header=True)
    
def pre_ESM_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    df = pd.read_csv(osp.join(root, '41592_2018_BFnmeth4633_MOESM3_ESM/Nathan_multi_mixed.csv'))
    df['sequence'] += gene.upper()
    df.to_csv(os.path.join(root, '41592_2018_BFnmeth4633_MOESM3_ESM/Nathan_multi_mixed_catATG.csv'), sep=',', index=False, header=True)

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
    
    promoters_strong = np.array([[strong[i], 1] for i in range(len(strong))])
    promoters_weak = np.array([[weak[i], 1] for i in range(len(weak))])
    promoters_non = np.array([[non[i], 0] for i in range(len(non))])
    promoters = np.concatenate([promoters_strong, promoters_weak, promoters_non], axis=0)
    df = pd.DataFrame(promoters, columns=['sequence', 'label'])
    df.to_csv(osp.join(RegulonDB_path, 'iPSW2L_PseKNC_mg1655.csv'), sep=',', index=False, header=True)

def pre_mg1655_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    df = pd.read_csv(osp.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655.csv'))
    df['sequence'] =  df['sequence'].str.upper() + 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(os.path.join(root, '1-s2.0-S088875431830613X-mmc1/iPSW2L_PseKNC_mg1655_catATG.csv'), sep=',', index=False, header=True)

def pre_csv_selected():
    if not os.path.exists(os.path.join(root, 'csvs/promoters/')):
        os.makedirs(os.path.join(root, 'csvs/promoters/'))
    
    path = '/hy-tmp/genes/'
    filenamelist = os.listdir(path)
    
    genes = []
    for i in range(len(filenamelist)):
        gene_path = osp.join(path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()

        for line in file:
            line = line.replace('\n', '').split(',')
            genes.append(line[2])

    filenamelist = os.listdir(osp.join(root, 'csvs/selected/'))
    for filename in filenamelist:
        # get real
        df_raw = pd.read_csv(osp.join(root, 'csvs/selected/', filename), delimiter=',')
        df_raw["label"] = -1
        
        for i in range(len(df_raw)):
            if type(df_raw.loc[i, 'PromoterSeq']) != str:
                continue
            df_raw.loc[i, 'label'] = 1
        df_real = df_raw[['PromoterSeq', 'label']]
        df_real = df_real[df_real['label'] != -1]
        promoters_real = df_real.to_numpy()

        # get fake
        cut_genes = []
        cut_length = len(promoters_real[-1][0])
        for i in range(len(promoters_real)):
            chosenID = random.randint(0, len(genes)-1)
            while (len(genes[chosenID])-200) < cut_length:
                chosenID = random.randint(0, len(genes)-1)
            start = random.randint(200, len(genes[chosenID]) - cut_length)
            cut_gene = genes[chosenID][start:start+cut_length]
            cut_genes.append([cut_gene, 0])
        promoters_fake = np.array(cut_genes)
        
        promoters = np.concatenate([promoters_real, promoters_fake], axis=0)
        df = pd.DataFrame(promoters, columns=['sequence', 'label'])
        df.to_csv(os.path.join(root, 'csvs/promoters/', filename), sep=',', index=False, header=True)
    
def pre_csv_selected_cat():
    if not os.path.exists(os.path.join(root, 'csvs/catATG/')):
        os.makedirs(os.path.join(root, 'csvs/catATG/'))

    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
        
    filenamelist = os.listdir(osp.join(root, 'csvs/promoters/'))
    for filename in filenamelist:
        df = pd.read_csv(osp.join(root, 'csvs/promoters/', filename), delimiter=',')
        df['sequence'] = df['sequence'].str.upper() + 'CTAACAGGAGGTGGACTAA' + gene.upper()
        df.to_csv(os.path.join(root, 'csvs/catATG', filename), sep=',', index=False, header=True)

def pre_ecos():
    # real
    f = open('/root/projects/DNABERT_Promotor/0pretrain/preprocess/genomes_filtered.csv')
    file = f.readlines()
    eco_names = [line.split(',')[1] for line in file][0:68]  # 68 ecos
    f.close()

    data_path = '/hy-tmp/genes/'
    filenamelist = [name for name in os.listdir(data_path) if name.split('.')[0] in eco_names]  # # 68 ecos' filenames

    promoters_real = []
    for filename in filenamelist:
        with open(osp.join(data_path, filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                promoter = line.replace('\n', '').split(',')[1]
                promoters_real.append([promoter, 1])
        f.close()
    promoters_real = np.array(promoters_real)

    # fake
    filenamelist = os.listdir(data_path)
    genes = []
    for i in range(len(filenamelist)):
        gene_path = osp.join(data_path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()

        for line in file:
            line = line.replace('\n', '').split(',')
            genes.append(line[2])
    
    cut_length = len(promoters_real[0][0])
    cut_genes = []
    for i in range(len(promoters_real)):
        chosenID = random.randint(0, len(genes)-1)
        while (len(genes[chosenID])-200) < cut_length:
            chosenID = random.randint(0, len(genes)-1)
        start = random.randint(200, len(genes[chosenID]) - cut_length)
        cut_gene = genes[chosenID][start:start+cut_length]
        cut_genes.append([cut_gene, 0])
    promoters_fake = np.array(cut_genes)
    promoters = np.concatenate([promoters_real, promoters_fake], axis=0)
    df = pd.DataFrame(promoters, columns=['sequence', 'label'])
    
    if not os.path.exists('/hy-tmp/promoter_real_fake_predict/'):
        os.makedirs('/hy-tmp/promoter_real_fake_predict/')
    df.to_csv('/hy-tmp/promoter_real_fake_predict/genes_ecos.csv', sep=',', index=False, header=True)

def pre_ecos_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    df = pd.read_csv('/hy-tmp/promoter_real_fake_predict/genes_ecos.csv')
    df['sequence'] =  df['sequence'].str.upper() + gene.upper()
    df.to_csv('/hy-tmp/promoter_real_fake_predict/genes_ecos_catATG.csv', sep=',', index=False, header=True)

def getTokensFromCSV():
    # gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    
    # model_name_or_path = '/root/pretrained/DNABERT-2-117M'
    model_name_or_path = '/root/pretrained/checkpoint-720000/'
    
    model_max_length = 100
    data_path = '/hy-tmp/promoter_real_fake_predict/'
    # token_path = '/hy-tmp/prokaryotes/eco_tokens_dnabert2/'
    # token_path = '/hy-tmp/promoter_real_fake_predict/genes_ecos/'
    
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    start = time.time()
    
    f = open(osp.join(data_path, 'genes_ecos_catATG.csv'))
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
    # np.savez(os.path.join(data_path, 'genes_ecos.npz'), **{'sequences':data, 'masks': masks, 'labels': labels})
    np.savez_compressed(os.path.join(data_path, 'genes_ecos_compressed.npz'), **{'sequences':data, 'masks': masks, 'labels': labels})
    print('saved.')
    # pdb.set_trace()
    print("Token saving time: ", time.time()-start, 's.')

def RegulonDB_read1(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            if line != ' \n':
                tmp = line.replace('\n', '').replace('\t', ' ').split(' ')
                tmp = tmp[-1].replace(' ', '')
                data.append(tmp)
    f.close()
    return data

def pre_MG1655_RegulonDB_promoter():
    # get real
    f = open(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB.csv'))
    file = f.readlines()[28:]
    f.close()
    # head = file[0].replace(' \n', '').split(',')
    lines = file[1:]
    
    promoters_real = []
    for line in lines:
        promoter = line.split(',')[5]
        if promoter == 'None':
            continue
        promoters_real.append([promoter, 1])
    promoters_real = np.array(promoters_real)
    
    # get fake
    path = '/hy-tmp/genes/'
    filenamelist = os.listdir(path)
    genes = []
    for i in range(len(filenamelist)):
        gene_path = osp.join(path, filenamelist[i])
        f = open(gene_path)
        file = f.readlines()
        f.close()
        for line in file:
            line = line.replace('\n', '').split(',')
            genes.append(line[2])
    # pdb.set_trace()
    
    cut_length = len(promoters_real[0][0])
    cut_genes = []
    for i in range(len(promoters_real)):
        chosenID = random.randint(0, len(genes)-1)
        while (len(genes[chosenID])-200) < cut_length:
            chosenID = random.randint(0, len(genes)-1)
        start = random.randint(200, len(genes[chosenID]) - cut_length)
        cut_gene = genes[chosenID][start:start+cut_length]            
        cut_genes.append([cut_gene, 0])
    promoters_fake = np.array(cut_genes)
    
    promoters = np.concatenate([promoters_real, promoters_fake], axis=0)
    df = pd.DataFrame(promoters, columns=['sequence', 'label'])
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters.csv'), sep=',', index=False, header=True)

def pre_MG1655_RegulonDB_promoter_cat():
    gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
    df = pd.read_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters.csv'))
    df['sequence'] = df['sequence'].str.upper() + 'CTAACAGGAGGTGGACTAA' + gene.upper()
    df.to_csv(osp.join(root, 'MG1655-RegulonDB/MG1655-RegulonDB_promoters_catATG.csv'), sep=',', index=False, header=True)

def pre_Escherichia_Corynebacterium():
    # Escherichia coli K-12 MG1655 T00007
    # Corynebacterium glutamicum ATCC 13032 (Kyowa Hakko) T00102
    root_path = '/data/whx/prokaryotes/genes/'
    save_path = '/data/whx/projects/DNABERT_Promotor/1train/promoter_generation/data/public_promoter/'
    gene_list = ['T00007', 'T00102']
    
    df_all = pd.DataFrame()
    for name in gene_list:
        file = pd.read_csv(osp.join(root_path, name+'.txt'), sep=',', header=None)
        df_all = pd.concat([df_all, file])
    df_all.columns = ['id', 'promoters', 'genes', 'complement', 'join']
    df_all.to_csv(osp.join(save_path, 'escherichia_corynebacterium.csv'), header=True, index=False)
        
    return


def pre_Escherichia_Vibrio():
    # Escherichia coli K-12 MG1655 T00007
    # Vibrio natriegens T04636
    root_path = '/data/whx/prokaryotes/genes/'
    save_path = '/data/whx/projects/DNABERT_Promotor/1train/promoter_generation/data/public_promoter/'
    gene_list = ['T00007', 'T04636']
    
    df_all = pd.DataFrame()
    for name in gene_list:
        file = pd.read_csv(osp.join(root_path, name+'.txt'), sep=',', header=None)
        df_all = pd.concat([df_all, file])
    df_all.columns = ['id', 'promoters', 'genes', 'complement', 'join']
    df_all.to_csv(osp.join(save_path, 'escherichia_vibrio.csv'), header=True, index=False)
    
    return
    
    
if __name__ == "__main__":
    # pre_ESM_promoter()
    # pre_ESM_cat()
    # pre_mg1655_promoter()
    # pre_mg1655_cat()
    # pre_csv_selected()
    # pre_csv_selected_cat()
    # pre_ecos()
    # pre_ecos_cat()
    # getTokensFromCSV()
    # pre_MG1655_RegulonDB_promoter()
    # pre_MG1655_RegulonDB_promoter_cat()
    
    # pre_Escherichia_Corynebacterium()
    # pre_Escherichia_Vibrio()
    
    pass
