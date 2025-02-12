import pdb
import os
import os.path as osp
import pandas as pd
import numpy as np
import subprocess


def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {command}")

def sort_promoters():
    root_path = '/root/projects/DNABERT_Promotor/'
    sequences = pd.read_csv(
        # osp.join(root_path, '2select/data/20240813_035617_ckpt72newtoken_steps1000_ecos_gen10000', 'gen_seqs.csv'), 
        # osp.join(root_path, '2select/data/20240911_125214_ckpt72newtoken_steps2000_ecos_gen10000', 'gen_seqs.csv'), 
        osp.join(root_path, '2select/data/20240912_043045_ckpt72newtoken_steps2000_ecos_gen100000', 'gen_seqs.csv'), 
        delimiter=',', 
        header=None
        )
    real_fake_predict = pd.read_csv(
        # osp.join(root_path, '2select/data/20240813_080246_ckpt72newtoken_ecos_epoch100_pred10000', 'real_fake_predict.csv'), 
        # osp.join(root_path, '2select/data/20240912_020551_ckpt72newtoken_ecos_epoch100_pred10000_steps2000', 'real_fake_predict.csv'), 
        osp.join(root_path, '2select/data/20240914_090831_ckpt72newtoken_ecos_epoch100_pred100000_steps2000', 'real_fake_predict.csv'), 
        delimiter=',', 
        header=None
        )
    cls2_activity = pd.read_csv(
        # osp.join(root_path, '2select/data/20240813_105012_ckpt72newtoken_ecos30_cls2_epoch100_pred10000_catATG', 'expression_intensity_classification.csv'), 
        # osp.join(root_path, '2select/data/20240912_021703_ckpt72newtoken_ecos30_cls2_epoch100_pred10000_catATG_steps2000', 'expression_intensity_classification.csv'), 
        osp.join(root_path, '2select/data/20240914_092058_ckpt72newtoken_ecos30_cls2_epoch100_pred100000_catATG_steps2000', 'expression_intensity_classification.csv'), 
        delimiter=',', 
        header=None
        )
    cls4_activity = pd.read_csv(
        # osp.join(root_path, '2select/data/20240813_145352_ckpt72newtoken_ecos30_cls4_epoch100_pred10000_catATG', 'expression_intensity_classification.csv'), 
        # osp.join(root_path, '2select/data/20240912_022004_ckpt72newtoken_ecos30_cls4_epoch100_pred10000_catATG_steps2000', 'expression_intensity_classification.csv'), 
        osp.join(root_path, '2select/data/20240914_091253_ckpt72newtoken_ecos30_cls4_epoch100_pred100000_catATG_steps2000', 'expression_intensity_classification.csv'), 
        delimiter=',', 
        header=None
        )
    cls2_strong_weak = pd.read_csv(
        # osp.join(root_path, '2select/data/20240813_144934_ckpt72newtoken_ecos30_cls2_epoch100_pred10000_catATG_strong_weak', 'expression_intensity_classification.csv'), 
        # osp.join(root_path, '2select/data/20240912_021841_ckpt72newtoken_ecos30_cls2_epoch100_pred10000_catATG_strong_weak_steps2000', 'expression_intensity_classification.csv'), 
        osp.join(root_path, '2select/data/20240914_102910_ckpt72newtoken_ecos30_cls2_epoch100_pred100000_catATG_strong_weak_steps2000', 'expression_intensity_classification.csv'), 
        delimiter=',', 
        header=None
        )
    
    df = pd.concat([sequences, real_fake_predict, cls2_activity, cls2_strong_weak, cls4_activity], axis=1)
    df.columns = ['sequence', 'real_fake_predict', 'cls2_activity', 'cls2_strong_weak', 'cls4_activity']
    
    df_sorted = df.sort_values(by=['cls4_activity', 'cls2_strong_weak', 'cls2_activity', 'real_fake_predict'], ascending=[False, False, False, False])
    # df_sorted.to_csv(osp.join(root_path, '2select/results', 'gen10000_fromlist30_catATG.csv'), sep=',', header=True)
    # df_sorted.to_csv(osp.join(root_path, '2select/results', 'gen10000_fromlist30_catATG_steps2000.csv'), sep=',', header=True)
    df_sorted.to_csv(osp.join(root_path, '2select/results', 'gen100000_fromlist30_catATG_steps2000.csv'), sep=',', header=True)
    # pdb.set_trace()

def merge():
    res_dir = '/root/projects/DNABERT_Promotor/2select/results/'
    filename = 'gen100000_fromlist30_catATG_steps2000'
    
    handle = open(osp.join(res_dir, filename+'.txt'))
    file = handle.readlines()
    handle.close()
    
    res_dict = {}
    for line in file:
        line_cut = line.replace('\n', '').split('\t')
        res_dict[line_cut[0]] = line
        
    handle2 = open(osp.join(res_dir, filename+'.csv'))
    file2 = handle2.readlines()
    handle2.close()

    f3 = open(osp.join(res_dir, filename+'_merge.csv'), 'w')
    f3.write(',sequence,real_fake_predict,cls2_activity,cls2_strong_weak,cls4_activity\n')
    for line in file2[1:]:
        line_cut = line.replace('\n', '').split(',')
        if line_cut[0] in res_dict.keys():
            newline = line.replace('\n', ',') + res_dict[line_cut[0]].replace('\t', ',')
        else:
            newline = line.replace('\n', ',,,,,,,,,,,,\n')
            # pdb.set_trace()
        f3.write(newline)
    f3.close()


def cutATG():
    dataroot = '/data/whx/2select/data/gen9/'
    # filename = 'ckpt72_steps2000_escherichia_gen100000'
    # filename = 'ckpt72new_steps2000_escherichia_gen100000'
    # filename = 'dnabert2_steps2000_escherichia_gen100000'
    # filename = 'ckpt72_steps2000_vibrio_gen100000'
    # filename = 'ckpt72new_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_vibrio_gen100000'
    # filename = 'ckpt72_steps2000_streptomyces_gen100000'
    filename = 'ckpt72new_steps2000_streptomyces_gen100000'
    # filename = 'dnabert2_steps2000_streptomyces_gen100000'
    

        
    handle = open(osp.join(dataroot, filename+'.csv'), 'r')    
    file = handle.readlines()
    handle.close()
    
    saveroot = '/data/whx/2select/data/gen9_cut/'
    f = open(osp.join(saveroot, filename+'_cut.csv'), 'w')
    # atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa
    # target_seq = 'ATGGTGAGCAAGGGCGAGGA'
    # target_seq = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGT'
    target_seq = 'ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGG'
    for seq in file:
        position = seq.find(target_seq)
        promoter = seq[:position]
        f.write(f'{promoter},{position}\n')
    f.close()

def csv2fasta():
    # dataroot = '/data/whx/2select/data/gen9/'
    dataroot = '/data/whx/2select/data/gen9_cut/'
    
    # filename = 'ckpt72_steps2000_escherichia_gen100000'
    # filename = 'ckpt72new_steps2000_escherichia_gen100000'
    # filename = 'dnabert2_steps2000_escherichia_gen100000'
    # filename = 'ckpt72_steps2000_vibrio_gen100000'
    # filename = 'ckpt72new_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_vibrio_gen100000'
    # filename = 'ckpt72_steps2000_streptomyces_gen100000'
    filename = 'ckpt72new_steps2000_streptomyces_gen100000'
    # filename = 'dnabert2_steps2000_streptomyces_gen100000'
    
    
    if '_cut' in dataroot:
        filename+='_cut'

    handle = open(osp.join(dataroot, filename+'.csv'), 'r')    
    file = handle.readlines()
    handle.close()
    
    
    f = open(osp.join(dataroot, filename+'.fasta'), 'w')
    
    # for line in file[1:]:
    #     line_cut = line.replace('\n', '').split(',')
    #     name = line_cut[0]
    #     seq = line_cut[1]
    #     f.write(f'>{name}\n')
    #     f.write(f'{seq}\n')
    
    if '_cut' in dataroot:
        number = 1
        for line in file:
            line_cut = line.split(',')[0]
            f.write(f'>{number}\n')
            f.write(f'{line_cut}\n')
            number += 1
    else:
        number = 1
        for line in file:
            line_cut = line.replace('\n', '')
            f.write(f'>{number}\n')
            f.write(f'{line_cut}\n')
            number += 1
        
    f.close()
    # pdb.set_trace()

def blast():
    # data_dir = '/data/whx/projects/DNABERT_Promotor/2select/data/'
    data_dir = '/data/whx/2select/data/'
    
    # res_dir = '/data/whx/projects/DNABERT_Promotor/2select/data/gen9/'
    res_dir = '/data/whx/2select/data/gen9_cut/'
    
    # fasta_path = osp.join(res_dir, 'gen100000_fromlist30_catATG_steps2000.fasta')
    # fasta_path = osp.join(res_dir, 'escherichia_vibrio_gen10000_steps2000.fasta')
    # fasta_path = osp.join(res_dir, 'escherichia_vibrio_gen100000_steps2000.fasta')
    
    # filename = 'ckpt72_steps2000_escherichia_gen100000'
    # filename = 'ckpt72new_steps2000_escherichia_gen100000'
    # filename = 'dnabert2_steps2000_escherichia_gen100000'
    # filename = 'ckpt72_steps2000_vibrio_gen100000'
    # filename = 'ckpt72new_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_vibrio_gen100000'
    # filename = 'ckpt72_steps2000_streptomyces_gen100000'
    # filename = 'ckpt72new_steps2000_streptomyces_gen100000'
    filename = 'dnabert2_steps2000_streptomyces_gen100000'
    
    
    if 'cut' in res_dir:
        filename += '_cut'
    fasta_path = osp.join(res_dir, filename+'.fasta')
    
    if 'escherichia' in filename:
        # Escherichia coli K-12 MG1655 T00007
        fna_path = osp.join(data_dir, 'fasta/GCF_000005845.2_ASM584v2_genomic.fna')
    elif 'streptomyces' in filename:
        # Streptomyces coelicolor T00085
        fna_path = osp.join(data_dir, 'fasta/GCF_000203835.1_ASM20383v1_genomic.fna')
    elif 'vibrio' in filename:
        # Vibrio natriegens T04636
        fna_path = osp.join(data_dir, 'fasta/GCF_001456255.1_ASM145625v1_genomic.fna')

        
    # run_command(f'/root/blast+/bin/makeblastdb -in {fna_path} -dbtype nucl')
    # run_command(f'/root/blast+/bin/blastn -query {fasta_path} -db {fna_path} -out {res_dir}/{filename}.txt -outfmt 6')
    
    # run_command(f'/home/whx/Downloads/ncbi-blast-2.16.0+/bin/makeblastdb -in {fna_path} -dbtype nucl')
    run_command(f'/home/whx/Downloads/ncbi-blast-2.16.0+/bin/blastn -query {fasta_path} -db {fna_path} -out {res_dir}/{filename}.txt -outfmt 6')

def blast_filter():
    # res_dir = '/data/whx/projects/DNABERT_Promotor/2select/results/'
    # res_dir = '/data/whx/projects/DNABERT_Promotor/2select/data/gen9/'
    res_dir = '/data/whx/2select/data/gen9_cut/'
    
    # filename = 'ckpt72_steps2000_escherichia_gen100000'
    # filename = 'ckpt72new_steps2000_escherichia_gen100000'
    # filename = 'dnabert2_steps2000_escherichia_gen100000'
    # filename = 'ckpt72_steps2000_vibrio_gen100000'
    # filename = 'ckpt72new_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_vibrio_gen100000'
    # filename = 'ckpt72_steps2000_streptomyces_gen100000'
    # filename = 'ckpt72new_steps2000_streptomyces_gen100000'
    filename = 'dnabert2_steps2000_streptomyces_gen100000'
    
    if 'cut' in res_dir:
        filename += '_cut'
    
    path = osp.join(res_dir, filename+'.txt')
    f = open(path)
    file = f.readlines()
    f.close()
    
    selected_list = []
    selected_lines = []
    for line in file:
        line_cut = line.replace('\n', '').split('\t')
        if not line_cut[0] in selected_list:
            selected_list.append(line_cut[0])
            selected_lines.append(line_cut)
        else:
            continue

    df = pd.DataFrame(selected_lines)
    df.columns = [
        'Query ID',
        'Subject ID',
        '% Identity',
        'Alignment Length',
        'Mismatches',
        'Gap Opens',
        'Query Start',
        'Query End',
        'Subject Start',
        'Subject End',
        'E-value',
        'Bit Score',
    ]
    df.to_csv(os.path.join(osp.join(res_dir, filename+'_blast.csv')), index=False, header=True)

def select_blast():
    # res_dir = '/data/whx/projects/DNABERT_Promotor/2select/results/'
    # res_dir = '/data/whx/projects/DNABERT_Promotor/2select/data/gen9/'
    res_dir = '/data/whx/2select/data/gen9_cut/'
    
    # # filename = 'gen100000_fromlist30_catATG_steps2000'
    # filename = 'escherichia_vibrio_gen100000_steps2000'

    filename = 'ckpt72_steps2000_escherichia_gen100000'
    # filename = 'ckpt72new_steps2000_escherichia_gen100000'
    # filename = 'dnabert2_steps2000_escherichia_gen100000'
    # filename = 'ckpt72_steps2000_vibrio_gen100000'
    # filename = 'ckpt72new_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_streptomyces_gen100000'
    
    # filename = 'ckpt72_steps2000_streptomyces_gen100000'
    # filename = 'ckpt72new_steps2000_streptomyces_gen100000'
    
    if 'cut' in res_dir:
        filename += '_cut'
        
    df_raw = pd.read_excel(
        os.path.join(osp.join(res_dir, filename+'.xlsx')),
        engine='openpyxl',
    )
    
    
    # path = osp.join(res_dir, 'escherichia_vibrio_gen100000_steps2000_blast_T00007.txt')
    # path = osp.join(res_dir, 'escherichia_vibrio_gen100000_steps2000_blast_T04636.txt')
    path = osp.join(res_dir, filename+'.txt')
    f = open(path)
    file = f.readlines()
    f.close()
    
    selected_list = []
    selected_lines = []
    for line in file:
        line_cut = line.replace('\n', '').split('\t')
        if not line_cut[0] in selected_list:
            selected_list.append(line_cut[0])
            newline = df_raw.iloc[int(line_cut[0])-1].tolist() + line_cut
            selected_lines.append(newline)
        else:
            continue

    df = pd.DataFrame(selected_lines)
    df.columns = np.concatenate([df_raw.columns, ['', '', '', '', '', '', '', '', '', '', '', '']])
    # df.to_csv(os.path.join(osp.join(res_dir, filename+'_blast_T00007.csv')), index=False, header=True)
    # df.to_csv(os.path.join(osp.join(res_dir, filename+'_blast_T04636.csv')), index=False, header=True)
    df.to_csv(os.path.join(osp.join(res_dir, filename+'_blast.csv')), index=False, header=True)


if __name__ == '__main__':
    # sort_promoters()
    # merge()
    
    
    # cutATG()
    # csv2fasta()
    # blast()
    blast_filter()
    
    
    
    
    # select_blast()
    
    pass
