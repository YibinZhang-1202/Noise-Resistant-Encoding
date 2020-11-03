import os
import pandas as pd
import random as rd

DATA_PATH = '../dataset/dirty/dirty_itunes_amazon_exp_data'
ORIGINAL_DATASET = 'test.csv'

NUM_TRIALS = 1
NUM_MISSPELLED_WORDS = [8,16,24,32,40]
NUM_MISSPELLED_CHAR_PER_WORD = 1
PERCENT_TUPLES_MISSPELLED = 0.1
# NUM_MISSPELLED_WORDS = [0]
# NUM_MISSPELLED_CHAR_PER_WORD = 0
# PERCENT_TUPLES_MISSPELLED = 0


def inject_misspelling(word, misspell_history):
    # print(word)

    if len(word) > NUM_MISSPELLED_CHAR_PER_WORD:
        misspell_type = rd.choice(['insertion', 'deletion', 'substitution', 'permutation'])
    else:
        misspell_type = 'insertion'

    if misspell_type == 'deletion' or misspell_type == 'substitution':
        misspell_char_index = rd.sample(set([i for i in range(len(word))]) - set(misspell_history), 1)[0]
        misspell_history.append(misspell_char_index)

        if misspell_type == 'deletion':
            word = word[0:misspell_char_index] + word[misspell_char_index + 1:]
        elif misspell_type == 'substitution':
            alphabet = set([chr(i) for i in range(ord('A'), ord('Z')+1)]).union(set([chr(i) for i in range(ord('a'), ord('z')+1)]))
            substitute_char = rd.sample(alphabet, k=1)[0]
            word = word[0:misspell_char_index] + str(substitute_char) + word[misspell_char_index + 1:]

    elif misspell_type == 'insertion':
        insert_index = rd.sample(set([i for i in range(len(word)+1)]) - set(misspell_history), 1)[0]
        alphabet = set([chr(i) for i in range(ord('A'), ord('Z') + 1)]).union(set([chr(i) for i in range(ord('a'), ord('z') + 1)]))
        insert_char = rd.sample(alphabet, k=1)[0]
        word = word[0:insert_index] + str(insert_char) + word[insert_index:]
        misspell_history.append(insert_index)

    elif misspell_type == 'permutation':
        permute_index = rd.sample(set([i for i in range(len(word)-1)]) - set(misspell_history), 1)[0]
        word = word[0:permute_index] + word[permute_index + 1] + word[permute_index] + word[permute_index + 2:]
        misspell_history.append(permute_index)


    # print(word)
    # print(misspell_history)
    return word, misspell_history


for g, x in enumerate(NUM_MISSPELLED_WORDS):

    for trial_i in range(0, NUM_TRIALS):
        out_f = 'test_misspelled_K' + str(NUM_MISSPELLED_CHAR_PER_WORD) + '-N' + str(x) + '-T' + str(trial_i) + '.csv'

        print(out_f)

        original_path = os.path.join(DATA_PATH, ORIGINAL_DATASET)
        original_f = pd.read_csv(original_path)
        original_f = original_f.astype(object)

        original_f = original_f.sample(frac=1).reset_index(drop=True)

        fields_to_modified = list(original_f)[2:]

        num_rows = PERCENT_TUPLES_MISSPELLED * len(original_f)

        for index, row in original_f.iterrows():
            print(index)
            if index > num_rows:
                break

            num_words_row = 0
            for field in fields_to_modified:
                num_words_row = num_words_row + len(str(row[field]).split(' '))


            misspelled_indices = rd.sample(set([i for i in range(num_words_row)]), min(x, num_words_row))

            col_index = 0
            for field in fields_to_modified:
                if row[field] == row[field]:
                    word_list = str(row[field]).split(' ')
                    # print(word_list)
                    for i in range(0, len(word_list)):
                        if col_index in misspelled_indices:
                            misspell_history = []
                            for j in range(NUM_MISSPELLED_CHAR_PER_WORD):
                                word_list[i], misspell_history = inject_misspelling(word_list[i], misspell_history)

                        col_index = col_index + 1

                    new_cell = ' '.join(map(str, word_list))
                    original_f._set_value(index, field, new_cell)

        original_f = original_f.sort_values(by='id')

        original_f.to_csv(os.path.join(DATA_PATH + '/original_misspelled_testsets', out_f), index=False)
