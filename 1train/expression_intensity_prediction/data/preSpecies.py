import os
import os.path as osp
import pandas as pd
import numpy as np
import pdb
import sys
import mygene
from bioservices import KEGG
from Bio import Entrez
Entrez.email = "walter_hx@163.com"
from Bio import SeqIO
from Bio.Seq import Seq
MAX_INT=sys.maxsize

root = '/root/projects/DNABERT_Promotor/1train/expression_intensity_prediction/data'
  
def get_mg1655_range():
    file = pd.read_excel(
        os.path.join(root, 'raw/1-s2.0-S2667370322000030-mmc1.xlsx'), 
        sheet_name='Supplementary data 3',
        engine='openpyxl',
        keep_default_na=False
    ).to_numpy()
    
    kegg = KEGG()

    mg1655_locus = file[2:,4]
    mg1655_locus = np.delete(mg1655_locus, np.where(mg1655_locus=='')[0])
    mg1655_intensity = file[2:,5]
    mg1655_intensity = np.delete(mg1655_intensity, np.where(mg1655_intensity=='')[0])
    mg1655_id = 'eco'

    start = 4356  # 560-2, 1844,1873-0,3145,3442-0,4050-0,4066,4082-0-b4629,4355-0-b4635
    f = open(os.path.join(root, 'species/mg1655_range.csv'), 'a')
    for i in range(start, len(mg1655_locus)):
        # print('mg1655 [{}]'.format(i), 'parsing gene locus:', mg1655_locus[i], 'intensity:', mg1655_intensity[i])
        gene_info = kegg.get(mg1655_id + ':' + mg1655_locus[i])  
        if gene_info == 403: 
            print('Error 403. Forbidden.')
            exit()
        elif gene_info == 404:
            print('Error 404. Can not get.')
            exit()
        seq = kegg.parse(gene_info).get("POSITION")
        line = str(i) + ',' + mg1655_locus[i] + ',' + seq + ',' + str(mg1655_intensity[i]) + '\n'
        f.write(line)
        print(line)
    f.close()

def get_mg1655_range_repair():
    '''
        为避免生成错误数据，丢弃这些信息，以下代码弃用
    '''
    file = pd.read_excel(
        os.path.join(root, 'raw/1-s2.0-S2667370322000030-mmc1.xlsx'), 
        sheet_name='Supplementary data 3',
        engine='openpyxl',
        keep_default_na=False
    ).to_numpy()
    
    mg = mygene.MyGeneInfo()
    
    mg1655_locus = file[2:,4]
    mg1655_locus = np.delete(mg1655_locus, np.where(mg1655_locus=='')[0])
    mg1655_intensity = file[2:,5]
    mg1655_intensity = np.delete(mg1655_intensity, np.where(mg1655_intensity=='')[0])

    f = open(os.path.join(root, 'species/mg1655_range.csv'), 'a')
    # 560-2, 1844, 1873-0, 3145, 3442-0, 4050-0, 4066, 4082-0-b4629, 4355-0-b4635
    for i in [560,1844,1873,3145,3442,4050,4066,4082,4355]:
        gene_info = mg.query(mg1655_locus[i], species='511145')
        print('##########', i, '##########')
        print(gene_info)
        # pdb.set_trace()
        if not gene_info['total'] == 0:
            for j in range(len(gene_info['hits'])):
                handle = Entrez.efetch(db="gene", id=gene_info['hits'][j]['_id'], rettype="fasta", retmode="text")
                seq_info = handle.read()
                print(seq_info)
                seq_info = seq_info.split('\n')
                seq = [s for s in seq_info if 'Annotation' in s][0]
                line = str(i)+'-'+gene_info['hits'][j]['symbol'] + ',' + mg1655_locus[i] + ',' + seq + ',' + str(mg1655_intensity[i]) + '\n'
                f.write(line)
            handle.close()
        else:
            line = str(i) + ',' + mg1655_locus[i] + ',,' + str(mg1655_intensity[i]) + '\n'
            f.write(line)
        print(line)
    f.close()

def cut_mg1655_promoter():
    genome_path = os.path.join(root, 'e_coli/MG1655.fasta')
    genome = [fa.seq for fa in SeqIO.parse(genome_path,  "fasta")][0]

    gene_csv_path = os.path.join(root, 'e_coli/mg1655_range.csv')
    f = open(gene_csv_path)
    csv = f.readlines()
    f.close()
    
    gene_path = os.path.join(root, 'e_coli/mg1655_genes.txt')
    f2 = open(gene_path, 'w')
    
    csv_filtered = []
    count = 0
    for line in csv:
        flag_join = ' '
        flag_complement = ' '
        
        gene_info = line.replace('\n', '').split(',')
        gene_id = gene_info[1]
        gene_position = gene_info[2]
        gene_intensity = gene_info[-1]
        
        if (not 'join' in gene_position) and (len(gene_info) == 4):   # 非拼接
            if ':' in gene_position:
                gene_position = gene_position.split(':')[-1]
            if '>' in gene_position:
                gene_position = gene_position.replace('>', '')
            if '<' in gene_position:
                gene_position = gene_position.replace('<', '')
            if not 'complement' in gene_position:  # 正链
                try:
                    startSeq, endSeq = gene_position.split('..')
                except:
                    continue
                startSeq = int(startSeq)
                endSeq = int(endSeq)
                startPromoter = startSeq - 200
                if startPromoter < 1:
                    continue
                gene = genome[startSeq-1:endSeq]
                promoter = genome[startPromoter-1:startSeq-1]
            else:  # 反链
                flag_complement = 'complement'
                try:
                    startSeq, endSeq = gene_position.split('(')[-1].split(')')[0].split('..')
                except:
                    continue
                startSeq = int(startSeq)
                endSeq = int(endSeq)
                endPromoter = endSeq + 200
                if endPromoter > len(genome):
                    continue
                gene = genome[startSeq-1:endSeq].complement()[::-1]
                promoter = genome[endSeq:endPromoter].complement()[::-1]
        else:
            flag_join = 'joined'   # gene是拼接的
            gene_position = ','.join(gene_info[2:-1])
            if ':' in gene_position:
                gene_position = gene_position.split(':')[-1]
            if '>' in gene_position:
                gene_position = gene_position.replace('>', '')
            if '<' in gene_position:
                gene_position = gene_position.replace('<', '')
            if not 'complement' in gene_position:  # 正链
                gene_cuts = gene_position.split('(')[-1].split(')')[0].split(',')
                gene_cuts = [[int(c) for c in cut.split('..')] for cut in gene_cuts]
                min_pos = MAX_INT
                gene = Seq('')
                for k in range(len(gene_cuts)):
                    try:
                        startSeq, endSeq = gene_cuts[k]
                    except:
                        continue
                    min_pos = min(startSeq, min_pos)
                    gene += genome[startSeq-1:endSeq]
                startPromoter = min_pos - 200
                if startPromoter < 1:
                    continue
                promoter = genome[startPromoter-1:min_pos-1]
            else:  # 反链
                flag_complement = 'complement'
                gene_cuts = gene_position.split('(')[-1].split(')')[0].split(',')
                gene_cuts = [[int(c) for c in cut.split('..')] for cut in gene_cuts]
                max_pos = 0
                gene = Seq('')
                for k in range(len(gene_cuts)):
                    try:
                        startSeq, endSeq = gene_cuts[k]
                    except:
                        continue
                    max_pos = max(endSeq, max_pos)
                    gene += genome[startSeq-1:endSeq]
                gene = gene.complement()[::-1]
                endPromoter = max_pos + 200
                if endPromoter > len(genome):
                    continue
                promoter = genome[max_pos:endPromoter]
                promoter = promoter.complement()[::-1]
        
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        csv_filtered.append(line.replace('\n', ''))
        newline = gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ',' + flag_join + ',' + gene_intensity + '\n'
        f2.write(newline)
        count += 1
    csv_filtered = np.array(csv_filtered)
    np.savetxt(os.path.join(root, 'e_coli/mg1655_filtered.csv'), csv_filtered, delimiter=",", fmt='%s')
    print('Writed. Length:{} Actually length:{}, minus:{}'.format(len(csv), count, len(csv)-count))
    f2.close()

    # pdb.set_trace()

def get_bl21_range():
    file1 = pd.read_excel(
        os.path.join(root, 'raw/1-s2.0-S2667370322000030-mmc1.xlsx'), 
        sheet_name='Supplementary data 3',
        engine='openpyxl',
        keep_default_na=False
    ).to_numpy()
    
    kegg = KEGG()
    
    bl21_locus = file1[2:,2]
    bl21_locus = np.delete(bl21_locus, np.where(bl21_locus=='')[0])
    bl21_intensity = file1[2:,3]
    bl21_intensity = np.delete(bl21_intensity, np.where(bl21_intensity=='')[0])
    bl21_id = 'ebl'
    # pdb.set_trace()
    
    start = 4094
    f = open(os.path.join(root, 'species/bl21_range.csv'), 'a')
    for i in range(start, len(bl21_locus)):
        print('bl21 [{}]'.format(i), 'parsing gene locus:', bl21_locus[i], 'intensity:', bl21_intensity[i])
        gene_info = kegg.get(bl21_id + ':' + bl21_locus[i])
        if gene_info == 403: 
            print('Error 403. Forbidden.')
            exit()
        seq = kegg.parse(gene_info).get("POSITION")
        print(seq)
        f.write(str(i) + ',' + bl21_locus[i] + ',' + seq + ',' + str(bl21_intensity[i]) + '\n')
    f.close()

def cut_bl21_promoter():
    genome_path = os.path.join(root, 'e_coli/BL21(DE3).fasta')
    genome = [fa.seq for fa in SeqIO.parse(genome_path,  "fasta")][0]
    
    gene_csv_path = os.path.join(root, 'e_coli/bl21_range.csv')
    f = open(gene_csv_path)
    csv = f.readlines()
    f.close()
    
    gene_path = os.path.join(root, 'e_coli/bl21_genes.txt')
    f2 = open(gene_path, 'w')
    
    csv_filtered = []
    count = 0
    for line in csv:
        flag_join = ' '
        flag_complement = ' '
        
        gene_info = line.replace('\n', '').split(',')
        gene_id = gene_info[1]
        gene_position = gene_info[2]
        gene_intensity = gene_info[-1]
        
        if (not 'join' in gene_position) and (len(gene_info) == 4):   # 非拼接
            if ':' in gene_position:
                gene_position = gene_position.split(':')[-1]
            if '>' in gene_position:
                gene_position = gene_position.replace('>', '')
            if '<' in gene_position:
                gene_position = gene_position.replace('<', '')
            if not 'complement' in gene_position:  # 正链
                try:
                    startSeq, endSeq = gene_position.split('..')
                except:
                    continue
                startSeq = int(startSeq)
                endSeq = int(endSeq)
                startPromoter = startSeq - 200
                if startPromoter < 1:
                    continue
                gene = genome[startSeq-1:endSeq]
                promoter = genome[startPromoter-1:startSeq-1]
            else:  # 反链
                flag_complement = 'complement'
                try:
                    startSeq, endSeq = gene_position.split('(')[-1].split(')')[0].split('..')
                except:
                    continue
                startSeq = int(startSeq)
                endSeq = int(endSeq)
                endPromoter = endSeq + 200
                if endPromoter > len(genome):
                    continue
                gene = genome[startSeq-1:endSeq].complement()[::-1]
                promoter = genome[endSeq:endPromoter].complement()[::-1]
        else:
            flag_join = 'joined'   # gene是拼接的
            gene_position = ','.join(gene_info[2:-1])
            if ':' in gene_position:
                gene_position = gene_position.split(':')[-1]
            if '>' in gene_position:
                gene_position = gene_position.replace('>', '')
            if '<' in gene_position:
                gene_position = gene_position.replace('<', '')
            if not 'complement' in gene_position:  # 正链
                gene_cuts = gene_position.split('(')[-1].split(')')[0].split(',')
                gene_cuts = [[int(c) for c in cut.split('..')] for cut in gene_cuts]
                min_pos = MAX_INT
                gene = Seq('')
                for k in range(len(gene_cuts)):
                    try:
                        startSeq, endSeq = gene_cuts[k]
                    except:
                        continue
                    min_pos = min(startSeq, min_pos)
                    gene += genome[startSeq-1:endSeq]
                startPromoter = min_pos - 200
                if startPromoter < 1:
                    continue
                promoter = genome[startPromoter-1:min_pos-1]
            else:  # 反链
                flag_complement = 'complement'
                gene_cuts = gene_position.split('(')[-1].split(')')[0].split(',')
                gene_cuts = [[int(c) for c in cut.split('..')] for cut in gene_cuts]
                max_pos = 0
                gene = Seq('')
                for k in range(len(gene_cuts)):
                    try:
                        startSeq, endSeq = gene_cuts[k]
                    except:
                        continue
                    max_pos = max(endSeq, max_pos)
                    gene += genome[startSeq-1:endSeq]
                gene = gene.complement()[::-1]
                endPromoter = max_pos + 200
                if endPromoter > len(genome):
                    continue
                promoter = genome[max_pos:endPromoter]
                promoter = promoter.complement()[::-1]
        
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        csv_filtered.append(line.replace('\n', ''))
        newline = gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ',' + flag_join + ',' + gene_intensity + '\n'
        f2.write(newline)
        count += 1
    csv_filtered = np.array(csv_filtered)
    np.savetxt(os.path.join(root, 'e_coli/bl21_filtered.csv'), csv_filtered, delimiter=",", fmt='%s')
    print('Writed. Length:{} Actually length:{}, minus:{}'.format(len(csv), count, len(csv)-count))
    f2.close()

    # pdb.set_trace()

def get_transcriptome_eco():
    name = 'EColi'
    short_name = 'eco'
    genome_name = 'GCF_000005845.2_ASM584v2_genomic.fna'
    
    datapath = osp.join(root, 'transcriptomes/raw', name)
    transcriptome_expression = pd.read_csv(osp.join(datapath, name+'_GeneExpression.csv'))
    transcriptome_expression['GeneId'] = transcriptome_expression['GeneId'].str[5:]
    
    transcriptome_raw = pd.read_csv(osp.join(datapath, short_name+'.tsv'), sep='\t')
    
    genome = [fa.seq for fa in SeqIO.parse(osp.join(datapath, genome_name),  "fasta")][0]
    
    gene_path = os.path.join(datapath, name+'_GeneExpression.txt')
    f = open(gene_path, 'w')
    
    for i in range(len(transcriptome_expression)):
        flag_complement = ' '
        gene_id = transcriptome_expression.iloc[i]['GeneId']
        try:
            idx = transcriptome_raw[transcriptome_raw['Locus tag']==gene_id].index[0]
        except:
            continue
        gene_info = transcriptome_raw.iloc[idx]
        gene_intensity = str(transcriptome_expression.iloc[i][name+'_FPKM'])
        
        if gene_info['Orientation'] == 'plus':  # 正链
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            startPromoter = startSeq - 200
            if startPromoter < 1:
                continue
            gene = genome[startSeq-1:endSeq]
            promoter = genome[startPromoter-1:startSeq-1]
        else:  # 反链
            flag_complement = 'complement'
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            endPromoter = endSeq + 200
            if endPromoter > len(genome):
                continue
            gene = genome[startSeq-1:endSeq].complement()[::-1]
            promoter = genome[endSeq:endPromoter].complement()[::-1]
            
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        
        newline = short_name+':'+gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ', ,' + gene_intensity + '\n'
        f.write(newline)
    f.close()
    return

def get_transcriptome_bsu():
    name = 'Bsub'
    short_name = 'bsu'
    genome_name = 'GCF_000009045.1_ASM904v1_genomic.fna'
    
    datapath = osp.join(root, 'transcriptomes/raw', name)
    transcriptome_expression = pd.read_csv(osp.join(datapath, name+'_GeneExpression.csv'))
    transcriptome_expression['GeneId'] = transcriptome_expression['GeneId'].str[5:]
    
    transcriptome_raw = pd.read_csv(osp.join(datapath, short_name+'.tsv'), sep='\t')
    
    genome = [fa.seq for fa in SeqIO.parse(osp.join(datapath, genome_name),  "fasta")][0]
    
    gene_path = os.path.join(datapath, name+'_GeneExpression.txt')
    f = open(gene_path, 'w')
    
    for i in range(len(transcriptome_expression)):
        flag_complement = ' '
        gene_id = transcriptome_expression.iloc[i]['GeneId']
        try:
            idx = transcriptome_raw[transcriptome_raw['Locus tag']==gene_id].index[0]
        except:
            continue
        gene_info = transcriptome_raw.iloc[idx]
        gene_intensity = str(transcriptome_expression.iloc[i][name+'_FPKM'])
        
        if gene_info['Orientation'] == 'plus':  # 正链
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            startPromoter = startSeq - 200
            if startPromoter < 1:
                continue
            gene = genome[startSeq-1:endSeq]
            promoter = genome[startPromoter-1:startSeq-1]
        else:  # 反链
            flag_complement = 'complement'
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            endPromoter = endSeq + 200
            if endPromoter > len(genome):
                continue
            gene = genome[startSeq-1:endSeq].complement()[::-1]
            promoter = genome[endSeq:endPromoter].complement()[::-1]
            
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        
        newline = short_name+':'+gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ', ,' + gene_intensity + '\n'
        f.write(newline)
    f.close()
    return

def get_transcriptome_cglu():
    name = 'Cglu'
    short_name = 'cglu'
    genome_name = 'GCF_000196335.1_ASM19633v1_genomic.fna'
    
    datapath = osp.join(root, 'transcriptomes/raw', name)
    transcriptome_expression = pd.read_csv(osp.join(datapath, name+'_GeneExpression.csv'))
    transcriptome_expression['GeneId'] = transcriptome_expression['GeneId'].str[5:]
    
    transcriptome_raw = pd.read_csv(osp.join(datapath, short_name+'.tsv'), sep='\t')
    
    genome = [fa.seq for fa in SeqIO.parse(osp.join(datapath, genome_name),  "fasta")][0]
    
    gene_path = os.path.join(datapath, name+'_GeneExpression.txt')
    f = open(gene_path, 'w')
    
    for i in range(len(transcriptome_expression)):
        flag_complement = ' '
        gene_id = transcriptome_expression.iloc[i]['GeneId']
        try:
            idx = transcriptome_raw[transcriptome_raw['Locus tag']==gene_id].index[0]
        except:
            continue
        gene_info = transcriptome_raw.iloc[idx]
        gene_intensity = str(transcriptome_expression.iloc[i][name+'_FPKM'])
        
        if gene_info['Orientation'] == 'plus':  # 正链
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            startPromoter = startSeq - 200
            if startPromoter < 1:
                continue
            gene = genome[startSeq-1:endSeq]
            promoter = genome[startPromoter-1:startSeq-1]
        else:  # 反链
            flag_complement = 'complement'
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            endPromoter = endSeq + 200
            if endPromoter > len(genome):
                continue
            gene = genome[startSeq-1:endSeq].complement()[::-1]
            promoter = genome[endSeq:endPromoter].complement()[::-1]
            
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        
        newline = short_name+':'+gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ', ,' + gene_intensity + '\n'
        f.write(newline)
    f.close()
    return

def get_transcriptome_vna():
    name = 'Vibr'
    short_name = 'vna'
    genome_name = 'GCF_001456255.1_ASM145625v1_genomic.fna'
    
    datapath = osp.join(root, 'transcriptomes/raw', name)
    transcriptome_expression = pd.read_csv(osp.join(datapath, name+'_GeneExpression.csv'))
    transcriptome_expression['GeneId'] = transcriptome_expression['GeneId'].str[5:]
    
    transcriptome_raw = pd.read_csv(osp.join(datapath, short_name+'.tsv'), sep='\t')
    
    genome = [fa.seq for fa in SeqIO.parse(osp.join(datapath, genome_name),  "fasta")][0]
    
    gene_path = os.path.join(datapath, name+'_GeneExpression.txt')
    f = open(gene_path, 'w')
    
    for i in range(len(transcriptome_expression)):
        flag_complement = ' '
        gene_id = transcriptome_expression.iloc[i]['GeneId']
        try:
            idx = transcriptome_raw[transcriptome_raw['Locus tag']==gene_id].index[0]
        except:
            continue
        gene_info = transcriptome_raw.iloc[idx]
        gene_intensity = str(transcriptome_expression.iloc[i][name+'_FPKM'])
        
        if gene_info['Orientation'] == 'plus':  # 正链
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            startPromoter = startSeq - 200
            if startPromoter < 1:
                continue
            gene = genome[startSeq-1:endSeq]
            promoter = genome[startPromoter-1:startSeq-1]
        else:  # 反链
            flag_complement = 'complement'
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            endPromoter = endSeq + 200
            if endPromoter > len(genome):
                continue
            gene = genome[startSeq-1:endSeq].complement()[::-1]
            promoter = genome[endSeq:endPromoter].complement()[::-1]
            
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        
        newline = short_name+':'+gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ', ,' + gene_intensity + '\n'
        f.write(newline)
    f.close()
    return

def get_transcriptome_lpl():
    name = 'WCFS1'
    short_name = 'lpl'
    genome_name = 'GCF_000203855.3_ASM20385v3_genomic.fna'
    
    datapath = osp.join(root, 'transcriptomes/raw', name)
    transcriptome_expression = pd.read_csv(osp.join(datapath, name+'_GeneExpression.csv'))
    transcriptome_expression['GeneId'] = transcriptome_expression['GeneId'].str[5:]
    
    transcriptome_raw = pd.read_csv(osp.join(datapath, short_name+'.tsv'), sep='\t')
    
    genome = [fa.seq for fa in SeqIO.parse(osp.join(datapath, genome_name),  "fasta")][0]
    
    gene_path = os.path.join(datapath, name+'_GeneExpression.txt')
    f = open(gene_path, 'w')
    
    for i in range(len(transcriptome_expression)):
        flag_complement = ' '
        gene_id = transcriptome_expression.iloc[i]['GeneId']
        try:
            idx = transcriptome_raw[transcriptome_raw['Locus tag']==gene_id].index[0]
        except:
            continue
        gene_info = transcriptome_raw.iloc[idx]
        gene_intensity = str(transcriptome_expression.iloc[i][name+'_FPKM'])
        
        if gene_info['Orientation'] == 'plus':  # 正链
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            startPromoter = startSeq - 200
            if startPromoter < 1:
                continue
            gene = genome[startSeq-1:endSeq]
            promoter = genome[startPromoter-1:startSeq-1]
        else:  # 反链
            flag_complement = 'complement'
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            endPromoter = endSeq + 200
            if endPromoter > len(genome):
                continue
            gene = genome[startSeq-1:endSeq].complement()[::-1]
            promoter = genome[endSeq:endPromoter].complement()[::-1]
            
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            continue
        
        newline = short_name+':'+gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ', ,' + gene_intensity + '\n'
        f.write(newline)
    f.close()
    return

def pre_fluorescence_1930():
    name = 'EColi'
    short_name = 'eco'
    genome_name = 'GCF_000005845.2_ASM584v2_genomic.fna'
    
    datapath = osp.join(root, 'Merged_TPM_and_Fluorescence')
    
    expression = pd.read_excel(osp.join(datapath, 'e-coli-promoter-data-0408.xlsx'), sheet_name='Data', engine='openpyxl')
    raw = pd.read_csv(osp.join(datapath, short_name+'.tsv'), sep='\t')
    genome = [fa.seq for fa in SeqIO.parse(osp.join(datapath, genome_name), "fasta")][0]
    
    gene_path = os.path.join(datapath, 'Fluorescence_Data.txt')
    f = open(gene_path, 'w')
    f.write('Id,Promoter,Gene,Flag_complement, ,Fluorescence\n')
    
    gene_path2 = os.path.join(datapath, 'Fluorescence_not_found.csv')
    not_found_list = []
    
    for i in range(len(expression)):
        flag_complement = ' '
        
        if expression.iloc[i]['Gene_Name'] in list(raw['Symbol']):
            idx = raw[raw['Symbol']==expression.iloc[i]['Gene_Name']].index[0]
        elif expression.iloc[i]['Gene_Name'] in list(raw['Locus tag']):
            idx = raw[raw['Locus tag']==expression.iloc[i]['Gene_Name']].index[0]
        else:
            print(expression.iloc[i]['Gene_Name'], 'not found.')
            not_found_list.append(i)
            continue
        gene_id = raw.loc[idx]['Locus tag']            
        gene_info = raw.iloc[idx]
        gene_fluorescence = str((expression.iloc[i]['T1'] + expression.iloc[i]['T2'] + expression.iloc[i]['T3'])/3)

        if gene_info['Orientation'] == 'plus':  # 正链
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            startPromoter = startSeq - 200
            if startPromoter < 1:
                print('start error.')
                continue
            gene = genome[startSeq-1:endSeq]
            promoter = genome[startPromoter-1:startSeq-1]
        else:  # 反链
            flag_complement = 'complement'
            startSeq = int(gene_info['Begin'])
            endSeq = int(gene_info['End'])
            endPromoter = endSeq + 200
            if endPromoter > len(genome):
                print('complement start error.')
                continue
            gene = genome[startSeq-1:endSeq].complement()[::-1]
            promoter = genome[endSeq:endPromoter].complement()[::-1]
            
        if not len(promoter) == (promoter.count('A') + promoter.count('T') + promoter.count('G') + promoter.count('C')):
            print('Promoter error!')
            continue
        if not len(gene) == (gene.count('A') + gene.count('T') + gene.count('G') + gene.count('C')):
            print('Gene error!')
            continue
        
        newline = short_name+':'+gene_id + ',' + str(promoter) + ',' + str(gene) + ',' + flag_complement + ', ,' + gene_fluorescence + '\n'
        f.write(newline)
    f.close()
    not_found_data = expression.iloc[not_found_list]
    not_found_data.to_csv(gene_path2)
    return

def pre_merge_tpm_fluorescence():
    file_fluorescence = pd.read_csv(osp.join(root, 'Merged_TPM_and_Fluorescence', 'Fluorescence_Data.txt'))
    file_fluorescence.columns = ['geneId', 'promoter', 'gene', 'complement', 'joined', 'fluorescence']
    
    file_tpm = pd.read_csv('/data/whx/expressions/transcript/tpm_gene/EColi_tpm_gene.txt')
    
    not_found_list = []
    newfile = []
    for i in range(len(file_fluorescence)):
        if file_fluorescence.iloc[i]['geneId'][4:] in list(file_tpm['geneId']):
            idx = file_tpm[file_tpm['geneId']==file_fluorescence.iloc[i]['geneId'][4:]].index[0]
            assert file_tpm.iloc[idx]['gene'] == file_fluorescence.iloc[i]['gene']
            assert file_tpm.iloc[idx]['promoter'] == file_fluorescence.iloc[i]['promoter']
            
            newline = [
                file_fluorescence.iloc[i]['geneId'],
                file_fluorescence.iloc[i]['promoter'],
                file_fluorescence.iloc[i]['gene'],
                file_fluorescence.iloc[i]['fluorescence'],
                file_tpm.iloc[idx]['tpm']
            ]
        else:
            not_found_list.append(i)
        newfile.append(newline)
    newfile = pd.DataFrame(newfile)
    newfile.columns = ['geneId', 'promoter', 'gene', 'fluorescence', 'tpm']
    newfile.to_csv(osp.join(root, 'Merged_TPM_and_Fluorescence', 'Merged_TPM_and_Fluorescence.txt'), index=False)
    # pdb.set_trace()
    
def pre_unique_tpm_fluorescence():
    file_tpm_fluorescence = pd.read_csv(osp.join(root, 'Merged_TPM_and_Fluorescence', 'Merged_TPM_and_Fluorescence.txt'))
        
    duplicated_ids = file_tpm_fluorescence[file_tpm_fluorescence.duplicated(subset='geneId', keep=False)]['geneId'].unique()
    average_df = file_tpm_fluorescence.groupby('geneId', as_index=False)['fluorescence'].mean()
    average_duplicates = average_df[average_df['geneId'].isin(duplicated_ids)]
    df_unique = file_tpm_fluorescence.drop_duplicates(subset='geneId', keep='first')
    
    for i in range(len(average_duplicates)):
        df_unique.loc[df_unique['geneId']==average_duplicates.iloc[i]['geneId'], 'fluorescence'] = average_duplicates.iloc[i]['fluorescence']
        
    df_unique.to_csv(osp.join(root, 'Merged_TPM_and_Fluorescence', 'Merged_TPM_and_Fluorescence_unique.txt'), index=False)

    
    file_tpm_fluorescence_csv = df_unique[['tpm', 'fluorescence']]
    file_tpm_fluorescence_csv.to_csv(osp.join(root, 'Merged_TPM_and_Fluorescence', 'file_tpm_fluorescence_unique.csv'), header=True, index=False)
    
    # pdb.set_trace()
    return


if __name__ == '__main__':
    # get_mg1655_range()
    # get_mg1655_range_repair()
    # cut_mg1655_promoter()
    # get_bl21_range()
    # cut_bl21_promoter()
    
    # get_transcriptome_eco()
    # get_transcriptome_bsu()
    # get_transcriptome_cglu()
    # get_transcriptome_vna()
    # get_transcriptome_lpl()
    
    # pre_fluorescence_1930()
    # pre_merge_tpm_fluorescence()
    # pre_unique_tpm_fluorescence()
    
    
    # pdb.set_trace()
    pass
