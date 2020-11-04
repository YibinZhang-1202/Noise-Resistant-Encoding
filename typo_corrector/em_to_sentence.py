import pandas as pd
import os

PATH = 'datasets/structured_itunes_amazon/ori_csv'
csv_files = [PATH+'/'+x for x in os.listdir(PATH)]

for x in csv_files:
    dataset = pd.read_csv(x)
    fields = dataset.columns[2:]

    gen_path = PATH.split('csv')[0] + 'txt'
    gen_path = gen_path + '/' + x.split('/')[-1].split('.')[0] + '.txt'
    print(gen_path)
    out_f = open(gen_path, 'w')

    for index, row in dataset.iterrows():
        left_line = ""
        right_line = ""
        for field in fields:
            if field.startswith('left_'):
                left_line = left_line + str(row[field]) + ' , '
            elif field.startswith('right_'):
                right_line = right_line + str(row[field]) + ' , '

        left_line = left_line[0:-3]
        right_line = right_line[0:-3]
        out_f.write(left_line+'\n')
        out_f.write(right_line+'\n')

    out_f.close()
