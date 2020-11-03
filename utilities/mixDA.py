import pandas as pd
import os
import random as rd


DATASET_PATH = 'dataset/dirty/dirty_itunes_amazon_exp_data'
ORIGINAL_DATASET = 'encoded_train_0.4-1.csv'
GENERATING_SET = 'encoded_train_0.4-1_aug.csv'

OPERATOR = ['span_del', 'span_shuffle', 'attr_del', 'attr_shuffle', 'entry_swap']
# OPERATOR = ['span_del', 'span_shuffle', 'entry_swap']

original_f = pd.read_csv(os.path.join(DATASET_PATH, ORIGINAL_DATASET))
original_f = original_f.astype(str)
out_f = pd.DataFrame(columns=original_f.columns)

left_fields = []
right_fields = []

for x in original_f.columns:
    if x.startswith('left'):
        left_fields.append(x)
    elif x.startswith('right'):
        right_fields.append(x)

for index, row in original_f.iterrows():
    curr_operator = rd.choice(OPERATOR)
    print(index)
    new_row = [row['id']] + [row['label']]

    if curr_operator == 'entry_swap':
        for a in right_fields:
            new_row = new_row + [row[a]]
        for a in left_fields:
            new_row = new_row + [row[a]]
    elif curr_operator == 'attr_del':
        delete_index = rd.choice([i for i in range(len(left_fields) + len(right_fields))])
        count = 0
        for a in left_fields:
            if count == delete_index:
                new_row = new_row + [None]
            else:
                new_row = new_row + [row[a]]
            count += 1
        for a in right_fields:
            if count == delete_index:
                new_row = new_row + [None]
            else:
                new_row = new_row + [row[a]]
            count += 1
    elif curr_operator == 'attr_shuffle':
        shuffle_index = rd.choice([i for i in range(len(left_fields) + len(right_fields) - 1)])
        for a in left_fields:
            new_row = new_row + [row[a]]
        for a in right_fields:
            new_row = new_row + [row[a]]
        temp = new_row[shuffle_index + 2]
        new_row[shuffle_index + 2] = new_row[shuffle_index + 3]
        new_row[shuffle_index + 3] = temp
    elif curr_operator == 'span_del':
        attr_index = rd.choice([i for i in range(len(left_fields) + len(right_fields))])
        count = 0
        for a in left_fields:
            if count == attr_index:
                attr_tokens = row[a].split(' ')
                if len(attr_tokens) == 0:
                    new_row = new_row + [row[a]]
                else:
                    delete_length = rd.choice([i for i in range(1, min(4,len(attr_tokens))+1)])
                    for b in range(delete_length):
                        delete_item = rd.choice(attr_tokens)
                        attr_tokens.remove(delete_item)
                    new_row = new_row + [' '.join(map(str,attr_tokens))]
            else:
                new_row = new_row + [row[a]]
            count += 1

        for a in right_fields:
            if count == attr_index:
                attr_tokens = row[a].split(' ')
                if len(attr_tokens) == 0:
                    new_row = new_row + [row[a]]
                else:
                    delete_length = rd.choice([i for i in range(1, min(4, len(attr_tokens)) + 1)])
                    for b in range(delete_length):
                        delete_item = rd.choice(attr_tokens)
                        attr_tokens.remove(delete_item)
                    new_row = new_row + [' '.join(map(str, attr_tokens))]
            else:
                new_row = new_row + [row[a]]
            count += 1
    elif curr_operator == 'span_shuffle':
        attr_index = rd.choice([i for i in range(len(left_fields) + len(right_fields))])
        count = 0
        for a in left_fields:
            if count == attr_index:
                attr_tokens = row[a].split(' ')
                if len(attr_tokens) <= 1:
                    new_row = new_row + [row[a]]
                else:
                    shuffle_index = rd.choice([i for i in range(0, len(attr_tokens)-1)])
                    sub_attr = attr_tokens[shuffle_index:shuffle_index+4]
                    rd.shuffle(sub_attr)
                    after_shuffle = attr_tokens[0:shuffle_index] + sub_attr + attr_tokens[shuffle_index+4:]
                    new_row = new_row + [' '.join(map(str, after_shuffle))]
            else:
                new_row = new_row + [row[a]]
            count += 1

        for a in right_fields:
            if count == attr_index:
                attr_tokens = row[a].split(' ')
                if len(attr_tokens) <= 1:
                    new_row = new_row + [row[a]]
                else:
                    shuffle_index = rd.choice([i for i in range(0, len(attr_tokens)-1)])
                    sub_attr = attr_tokens[shuffle_index:shuffle_index+4]
                    rd.shuffle(sub_attr)
                    after_shuffle = attr_tokens[0:shuffle_index] + sub_attr + attr_tokens[shuffle_index+4:]
                    new_row = new_row + [' '.join(map(str, after_shuffle))]
            else:
                new_row = new_row + [row[a]]
            count += 1
    # print(new_row)
    out_f.loc[index] = new_row

out_f.to_csv(os.path.join(DATASET_PATH, GENERATING_SET), index=False)
