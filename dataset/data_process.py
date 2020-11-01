import os
import pandas as pd

PATH = './dirty/dirty_itunes_amazon_exp_data1'
table_a_path = os.path.join(PATH, 'tableA.csv')
table_b_path = os.path.join(PATH, 'tableB.csv')
relation_path = os.path.join(PATH, 'train.csv')
out_path = os.path.join(PATH, 'train_processed.csv')


table_a_f = pd.read_csv(table_a_path)
table_b_f = pd.read_csv(table_b_path)
relation_f = pd.read_csv(relation_path)

# Set attribute fields
out_f_columns = table_a_f.columns.delete(0)
out_f_columns = out_f_columns.append(out_f_columns)

for i in range(0, out_f_columns.size):
    curr_index = str(out_f_columns[i])
    out_f_columns = out_f_columns.delete(i)
    if i < out_f_columns.size / 2:
        out_f_columns = out_f_columns.insert(i, 'left_'+curr_index)
    else:
        out_f_columns = out_f_columns.insert(i, 'right_' + curr_index)

out_f_columns = out_f_columns.insert(0, 'label')
out_f_columns = out_f_columns.insert(0, 'id')

out_f = pd.DataFrame(columns=out_f_columns)


# Load data
for index, row in relation_f.iterrows():
    left = table_a_f.loc[table_a_f['id'] == row['ltable_id']]
    left = list(left.iloc[0])
    # left = list(table_a_f.loc[row['ltable_id']])
    left.pop(0)
    right = table_b_f.loc[table_b_f['id'] == row['rtable_id']]
    right = list(right.iloc[0])
    # right = list(table_b_f.loc[row['rtable_id']])
    right.pop(0)
    out_f.loc[index] = [index] + [row['label']] + left + right
    print([index] + [row['label']] + left + right)

out_f.to_csv(out_path,index=False)
