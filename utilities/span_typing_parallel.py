import spacy
import pandas as pd
import os
import multiprocessing as mp
import numpy as np
from spacy import displacy

DATASET_PATH = 'dataset/dirty/dirty_itunes_amazon_exp_data'
ORIGINAL_DATASET = 'train_aug.csv'
GENERATED_SET = 'train_aug_snst.csv'

NUM_CORES = 1

def process_chunk_function(chunk):
    special_tokens = set()
    chunk = chunk.astype(object)
    fields = chunk.columns[2:]

    for index, row in chunk.iterrows():
        for field in fields:
            doc = nlp(str(row[field]))

            new_cell = str(row[field])
            # print(new_cell)

            for ent in tuple(reversed(doc.ents)):
                if len(ent.text.split(' ')) != 1:
                    new_cell = new_cell[0:ent.end_char] + " [\\" + ent.label_ + "]" + new_cell[ent.end_char:]
                    special_tokens.add("[" + "\\" + ent.label_ + "]")
                new_cell = new_cell[0:ent.start_char] + "[" + ent.label_ + "] " + new_cell[ent.start_char:]
                special_tokens.add("[" + ent.label_ + "]")
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)

            chunk._set_value(index, field, new_cell)
            # displacy.serve(doc, style="ent")
            # print(new_cell)
            # print()
        print("Row ", index, " finished.")
    return chunk, special_tokens


def span_typing_exec(path, ori_set, gen_set):
    global nlp

    nlp = spacy.load("en_core_web_sm")

    dataset = pd.read_csv(os.path.join(path, ori_set))
    df_chunks = np.array_split(dataset, NUM_CORES)

    all_special_tokens = set()

    with mp.Pool(NUM_CORES) as pool:
        processes_results = pool.map(process_chunk_function, df_chunks)

        chunk_list = []

        for x in processes_results:
            chunk_list.append(x[0])
            all_special_tokens = all_special_tokens.union(x[1])

        processed_df = pd.concat(chunk_list, ignore_index=True)

    processed_df.to_csv(os.path.join(path, gen_set), index=False)

    return all_special_tokens


if __name__ == "__main__":
    all_special_tokens = span_typing_exec(DATASET_PATH, ORIGINAL_DATASET, GENERATED_SET)
    print(all_special_tokens)
