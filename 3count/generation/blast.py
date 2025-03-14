import os.path as osp
import numpy as np
import pandas as pd
import pdb


def getBlastMatrix():
    # path = '/data/whx/2select/data/gen9/'
    path = '/data/whx/2select/data/gen9_cut/'
    
    # filename = 'ckpt72_steps2000_escherichia_gen100000'
    # filename = 'ckpt72_steps2000_streptomyces_gen100000'
    # filename = 'ckpt72_steps2000_vibrio_gen100000'
    # filename = 'ckpt72new_steps2000_escherichia_gen100000'
    # filename = 'ckpt72new_steps2000_streptomyces_gen100000'
    # filename = 'ckpt72new_steps2000_vibrio_gen100000'
    # filename = 'dnabert2_steps2000_escherichia_gen100000'
    # filename = 'dnabert2_steps2000_streptomyces_gen100000'
    filename = 'dnabert2_steps2000_vibrio_gen100000'


    if 'cut' in path:
        filename += '_cut'
    
    df = pd.read_csv(osp.join(path, filename+'.csv'), header=None)
    df.columns = ['seq', 'start_codon']    
    
    df_blast = pd.read_csv(osp.join(path, filename+'_blast.csv'))
    df_blast.columns = [
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
    
    count_list = {
        'complete_match': 0,
        'mutation': 0,
        'gen_mutation': 0,
        'gen': 0,
    }
    
    for i in range(len(df_blast)):
        promoter_end = int(df.iloc[int(df_blast.iloc[i]['Query ID'])-1]['start_codon'])
        query_start = int(df_blast.iloc[i]['Query Start'])
        query_end = int(df_blast.iloc[i]['Query End'])
        
        match_rate = float(df_blast.iloc[i]['% Identity']) / 100
        if match_rate == 1:
            if (promoter_end - query_start) > 100:
                count_list['complete_match'] += 1
            else:
                count_list['mutation'] += 1
        else:
            count_list['gen_mutation'] += 1
    count_list['gen'] = 100000 - len(df_blast)
    
    print(count_list)
    

if __name__ == '__main__':
    getBlastMatrix()
    pass
