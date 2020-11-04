from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import numpy as np
import sys
import logging


DATASET_PATH = 'dataset'
MODEL_PATH = 'model_state'
TEST_SET_DIR = sys.argv[1]
BERT_MODEL = sys.argv[2]
CLASSIFIER_MODEL = sys.argv[3]
LOG_NAME = sys.argv[4]

TEST_SPECIAL_TOKENS = {'[\\ORG]', '[GPE]', '[TIME]', '[EVENT]', '[NORP]', '[ORG]', '[ORDINAL]', '[PERSON]', '[\\TIME]', '[LOC]', '[\\PERSON]', '[\\QUANTITY]', '[\\NORP]', '[\\LAW]', '[LAW]', '[MONEY]', '[PRODUCT]', '[\\CARDINAL]', '[QUANTITY]', '[CARDINAL]', '[LANGUAGE]', '[\\PERCENT]', '[\\GPE]', '[DATE]', '[PERCENT]', '[\\MONEY]', '[\\WORK_OF_ART]', '[\\EVENT]', '[\\LOC]', '[\\FAC]', '[FAC]', '[\\PRODUCT]', '[\\LANGUAGE]', '[WORK_OF_ART]', '[\\ORDINAL]', '[\\DATE]'}

WRITE_FALSE_INDICES = True

# Using token_length = 512 for Textual Company dataset and 256 for others
token_length = 256

logging.basicConfig(level=logging.CRITICAL, handlers=[logging.FileHandler(LOG_NAME),logging.StreamHandler()])

logging.critical(TEST_SET_DIR)
logging.critical(sys.argv)


batch_size = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'


######################################################################################################################
class Ditto_linear(BertModel):
    def __init__(self, config):
        super(Ditto_linear, self).__init__(config)


class Linear_classifier(nn.Module):
    def __init__(self):
        super(Linear_classifier, self).__init__()
        the_bert = torch.load(os.path.join(MODEL_PATH, BERT_MODEL))
        self.dropout = nn.Dropout(the_bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(the_bert.config.hidden_size, 2)
        # self.classifier = nn.Sequential(
        #     nn.Linear(model_finetune_bert.config.hidden_size, 2 * model_finetune_bert.config.hidden_size),
        #     nn.BatchNorm1d(2 * model_finetune_bert.config.hidden_size),
        #     nn.Tanh(),
        #     # nn.Linear(2 * model_finetune_bert.config.hidden_size, model_finetune_bert.config.hidden_size),
        #     # nn.ReLU(),
        #     nn.Linear(2 * model_finetune_bert.config.hidden_size, model_finetune_bert.config.num_labels)
        # )

    def forward(self, x):
        outputs = self.dropout(x)
        outputs = self.classifier(outputs)

        return outputs


def load_data(test_set, model_finetune_bert, tokenizer):
    # Load dataset
    fields = test_set.columns[2:]
    test_labels = test_set.label.values
    test_inputs = test_set[fields].values

    # Add special tokens
    special_tokens_list = {'[COL]', '[VAL]'}
    test_special_tokens = TEST_SPECIAL_TOKENS

    special_tokens_list = special_tokens_list.union(test_special_tokens)
    special_tokens_dict = {'additional_special_tokens': list(special_tokens_list)}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_tokens(fields.to_list())
    model_finetune_bert.resize_token_embeddings(len(tokenizer))

    # Serialize inputs
    test_inputs_ids = []
    test_attention_masks = []
    test_token_type_ids = []

    for the_input in test_inputs:
        left = ''
        right = ''
        for i, v in enumerate(the_input):
            if fields[i].startswith('left'):
                left = left + ' [COL] ' + fields[i] + ' [VAL] ' + str(v)
            elif fields[i].startswith('right'):
                right = right + ' [COL] ' + fields[i] + ' [VAL] ' + str(v)

        encoded_input = tokenizer.encode_plus(left, right, add_special_tokens=True, max_length=token_length,
                                              pad_to_max_length=True)

        test_inputs_ids.append(encoded_input['input_ids'])
        test_attention_masks.append(encoded_input['attention_mask'])
        test_token_type_ids.append(encoded_input['token_type_ids'])

    test_inputs_ids = torch.tensor(test_inputs_ids)
    test_labels = torch.tensor(test_labels)
    test_attention_masks = torch.tensor(test_attention_masks)
    test_token_type_ids = torch.tensor(test_token_type_ids)

    # Create the DataLoader
    test_data = TensorDataset(test_inputs_ids, test_attention_masks, test_labels, test_token_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, shuffle=False)

    return test_dataloader


def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_finetune_bert = torch.load(os.path.join(MODEL_PATH, BERT_MODEL))
    classifier = torch.load(os.path.join(MODEL_PATH, CLASSIFIER_MODEL))

    return tokenizer, model_finetune_bert, classifier


def flat_accuracy(preds, labels, set_index):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # print(pred_flat)
    # print(labels_flat)
    false_indices = np.argwhere(np.not_equal(pred_flat, labels_flat) == True).flatten() + set_index
    return np.sum(pred_flat == labels_flat) / len(labels_flat), false_indices


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def update_stats(tps, tns, fps, fns, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for i in range(0, len(labels_flat)):
        if pred_flat[i] == 0:
            if labels_flat[i] == pred_flat[i]:
                tns = tns + 1
            elif labels_flat[i] != pred_flat[i]:
                fns = fns + 1
        elif pred_flat[i] == 1:
            if labels_flat[i] == pred_flat[i]:
                tps = tps + 1
            elif labels_flat[i] != pred_flat[i]:
                fps = fps +1

    return tps, tns, fps, fns


def precision(tps, tns, fps, fns):
    return 100 * tps / max(tps + fps, 1)


def recall(tps, tns, fps, fns):
    return 100 * tps / max(tps + fns, 1)


def f1(precision, recall):
    return 2 * precision * recall / max(precision + recall, 1)


def eval_testdata(bert, classifier, test_dataloader, file_name, experiment_results):
    logging.critical("Running Validation...")

    t0 = time.time()
    bert.eval()
    classifier.eval()
    bert.to(device)
    classifier.to(device)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    tps, tns, fps, fns = 0, 0, 0, 0

    set_index = 0
    false_indices = []

    for batch in test_dataloader:
        batch = tuple(t for t in batch)
        b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)
        b_token_type_ids = b_token_type_ids.to(device)

        with torch.no_grad():
            outputs = bert(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
            logits = classifier(outputs[1])

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy, batch_false_indices = flat_accuracy(logits, label_ids, set_index)
        false_indices.extend(batch_false_indices)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

        tps, tns, fps, fns = update_stats(tps, tns, fps, fns, logits, label_ids)

        set_index += batch_size

    the_precision = precision(tps, tns, fps, fns)
    the_recall = recall(tps, tns, fps, fns)
    the_f1 = f1(the_precision, the_recall)

    if WRITE_FALSE_INDICES:
        with open('false_indices', 'w') as out_files:
            out_files.write(' '.join(map(str, false_indices)))

    # print(false_indices)
    logging.critical("{:}".format(file_name))
    logging.critical("Precision: {:} | Recall: {:} | F1: {:}".format(the_precision, the_recall, the_f1))
    logging.critical("TPS: {:} | TNS: {:} | FPS: {:} | FNS: {:}".format(tps, tns, fps, fns))
    logging.critical("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    logging.critical("  Validation took: {:}".format(format_time(time.time() - t0)))
    logging.critical("")
    logging.critical("Test complete!")

    this_result = {'File_name': file_name, 'F1-score': the_f1, 'Precision': the_precision, 'Recall': the_recall, 'TPS': tps, 'TNS': tns, 'FPS': fps, 'FNS': fns}
    experiment_results = experiment_results.append(this_result, ignore_index=True)

    return experiment_results
######################################################################################################################

experiment_results = pd.DataFrame(columns=['File_name', 'F1-score', 'Precision', 'Recall', 'TPS', 'TNS', 'FPS', 'FNS'])


for f in os.listdir(os.path.join(DATASET_PATH, TEST_SET_DIR)):

    if f.endswith(".csv"):

        test_set = pd.read_csv(os.path.join(DATASET_PATH, os.path.join(TEST_SET_DIR, f)))

        tokenizer, bert, classifier = load_model()

        test_dataloader = load_data(test_set=test_set, model_finetune_bert=bert, tokenizer=tokenizer)

        experiment_results = eval_testdata(bert=bert, classifier=classifier, test_dataloader=test_dataloader, file_name=f, experiment_results=experiment_results)


experiment_results.to_csv("experiment_results" + TEST_SET_DIR.split('/')[1] + ".csv")
