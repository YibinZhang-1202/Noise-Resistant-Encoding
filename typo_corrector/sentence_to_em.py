import os
import pandas as pd
import random

PATH = 'datasets/structured_itunes_amazon'

ori_testset = pd.read_csv(os.path.join(PATH,'test_50.csv'))

label = ori_testset['label']
fields = ori_testset.columns

txt_files = [PATH+'/pred_txt/'+x for x in os.listdir(PATH+'/pred_txt')]

for t in txt_files:
    df = pd.DataFrame(columns=fields)
    txt_f = open(t,'r')
    txt_lines = txt_f.readlines()
    txt_lines_index = 0

    for i,x in enumerate(label):
        left_tuple = txt_lines[txt_lines_index].split(',')
        left_tuple = [x.strip() for x in left_tuple]
        txt_lines_index = txt_lines_index + 1
        right_tuple = txt_lines[txt_lines_index].split(',')
        right_tuple = [x.strip() for x in right_tuple]
        txt_lines_index = txt_lines_index + 1

        while len(left_tuple) > (len(fields)-2)/2:
            # print(left_tuple)
            pos = random.randint(0,len(left_tuple)-2)
            left_tuple[pos] = left_tuple[pos] + ' ' + left_tuple.pop(pos+1)
            # print(left_tuple)

        while len(right_tuple) > (len(fields)-2)/2:
            # print(right_tuple)
            pos = random.randint(0,len(right_tuple)-2)
            right_tuple[pos] = right_tuple[pos] + ' ' + right_tuple.pop(pos + 1)
            # print(right_tuple)

        df.loc[i] = [i] + [x] + left_tuple + right_tuple

    gen_path = t.replace('txt','csv')
    print(gen_path)
    df.to_csv(gen_path,index=False)
    txt_f.close()

