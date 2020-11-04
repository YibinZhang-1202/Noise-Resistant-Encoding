from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel, \
    get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
import datetime
import sys
import logging


DATASET_PATH = 'dataset'
MODEL_PATH = 'model_state'
TRAIN_SET = sys.argv[1]
AUG_SET = sys.argv[2]
VALIDATION_SET = sys.argv[1].split('/')[0] + '/' + sys.argv[1].split('/')[1] + '/' + sys.argv[6]
BERT_NAME = sys.argv[3]
CLASSIFIER_NAME = sys.argv[4]

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("training_log.log"),logging.StreamHandler()])

logging.info(sys.argv)

batch_size = 8
# Using token_length = 512 for Textual Company dataset and 256 for others
token_length = 256
# default 3e-5
learning_rate = 3e-5

epochs = int(sys.argv[5])


# Ditto's model architecture
class Ditto_linear(BertModel):
    def __init__(self, config):
        super(Ditto_linear, self).__init__(config)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model_finetune_bert = Ditto_linear.from_pretrained("bert-base-uncased", output_attentions=False,
                                                   output_hidden_states=True, )
# model_finetune_bert = torch.load(os.path.join(MODEL_PATH, 's_itunes_amazon/s_ia_full_bert.pth'))

class Linear_classifier(nn.Module):
    def __init__(self):
        super(Linear_classifier, self).__init__()
        self.dropout = nn.Dropout(model_finetune_bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(model_finetune_bert.config.hidden_size, 2)
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


linear_classifier = Linear_classifier()
# linear_classifier = torch.load(os.path.join(MODEL_PATH, 's_itunes_amazon/s_ia_full_classifier.pth'))

# Load dataset
train_set = pd.read_csv(os.path.join(DATASET_PATH, TRAIN_SET))
aug_set = pd.read_csv(os.path.join(DATASET_PATH, AUG_SET))
validation_set = pd.read_csv(os.path.join(DATASET_PATH, VALIDATION_SET))

train_labels = train_set.label.values
validation_labels = validation_set.label.values

fields = train_set.columns[2:]
train_inputs = train_set[fields].values
aug_inputs = aug_set[fields].values
validation_inputs = validation_set[fields].values

# Add special tokens
special_tokens_list = {'[COL]', '[VAL]'}

training_special_tokens = {'[\\ORG]', '[\\PERSON]', '[\\TIME]', '[\\MONEY]', '[\\PRODUCT]', '[\\LANGUAGE]', '[WORK_OF_ART]', '[TIME]', '[\\PERCENT]', '[LAW]', '[GPE]', '[ORDINAL]', '[\\LOC]', '[\\GPE]', '[PERSON]', '[EVENT]', '[NORP]', '[\\ORDINAL]', '[\\EVENT]', '[PERCENT]', '[LOC]', '[\\QUANTITY]', '[PRODUCT]', '[QUANTITY]', '[\\FAC]', '[\\WORK_OF_ART]', '[MONEY]', '[CARDINAL]', '[\\NORP]', '[ORG]', '[\\DATE]', '[\\LAW]', '[FAC]', '[\\CARDINAL]', '[LANGUAGE]', '[DATE]'}

training_aug_special_tokens = {'[\\ORG]', '[\\PERSON]', '[\\TIME]', '[\\MONEY]', '[\\PRODUCT]', '[\\LANGUAGE]', '[WORK_OF_ART]', '[TIME]', '[\\PERCENT]', '[LAW]', '[GPE]', '[ORDINAL]', '[\\LOC]', '[\\GPE]', '[PERSON]', '[EVENT]', '[NORP]', '[\\ORDINAL]', '[\\EVENT]', '[PERCENT]', '[LOC]', '[\\QUANTITY]', '[PRODUCT]', '[QUANTITY]', '[\\FAC]', '[\\WORK_OF_ART]', '[MONEY]', '[CARDINAL]', '[\\NORP]', '[ORG]', '[\\DATE]', '[\\LAW]', '[FAC]', '[\\CARDINAL]', '[LANGUAGE]', '[DATE]'}

validation_special_tokens = {'[\\ORG]', '[\\PERSON]', '[\\TIME]', '[\\MONEY]', '[\\PRODUCT]', '[\\LANGUAGE]', '[WORK_OF_ART]', '[TIME]', '[\\PERCENT]', '[LAW]', '[GPE]', '[ORDINAL]', '[\\LOC]', '[\\GPE]', '[PERSON]', '[EVENT]', '[NORP]', '[\\ORDINAL]', '[\\EVENT]', '[PERCENT]', '[LOC]', '[\\QUANTITY]', '[PRODUCT]', '[QUANTITY]', '[\\FAC]', '[\\WORK_OF_ART]', '[MONEY]', '[CARDINAL]', '[\\NORP]', '[ORG]', '[\\DATE]', '[\\LAW]', '[FAC]', '[\\CARDINAL]', '[LANGUAGE]', '[DATE]'}

special_tokens_list = special_tokens_list.union(training_special_tokens).union(validation_special_tokens).union(training_aug_special_tokens)
special_tokens_dict = {'additional_special_tokens': list(special_tokens_list)}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.add_tokens(fields.to_list())


if torch.cuda.device_count() > 1 and device == 'cuda':
    model_finetune_bert = nn.DataParallel(model_finetune_bert)
    linear_classifier = nn.DataParallel(linear_classifier)
    model_finetune_bert.module.resize_token_embeddings(len(tokenizer))
    logging.info('Training on {:} GPUs'.format(torch.cuda.device_count()))
else:
    model_finetune_bert.resize_token_embeddings(len(tokenizer))


# Serialize inputs
train_input_ids = []
aug_input_ids = []
validation_input_ids = []
train_attention_masks = []
aug_attention_masks = []
validation_attention_masks = []
train_token_type_ids = []
aug_token_type_ids = []
validation_token_type_ids = []


all_set = [train_inputs, aug_inputs, validation_inputs]
for the_set in all_set:
    for the_input in the_set:
        left = ''
        right = ''
        for i, v in enumerate(the_input):
            if fields[i].startswith('left'):
                left = left + ' [COL] ' + fields[i] + ' [VAL] ' + str(v)
            elif fields[i].startswith('right'):
                right = right + ' [COL] ' + fields[i] + ' [VAL] ' + str(v)

        encoded_input = tokenizer.encode_plus(left, right, add_special_tokens=True, max_length=token_length, pad_to_max_length=True)

        if the_set is train_inputs:
            train_input_ids.append(encoded_input['input_ids'])
            train_attention_masks.append(encoded_input['attention_mask'])
            train_token_type_ids.append(encoded_input['token_type_ids'])
        elif the_set is aug_inputs:
            aug_input_ids.append(encoded_input['input_ids'])
            aug_attention_masks.append(encoded_input['attention_mask'])
            aug_token_type_ids.append(encoded_input['token_type_ids'])
        elif the_set is validation_inputs:
            validation_input_ids.append(encoded_input['input_ids'])
            validation_attention_masks.append(encoded_input['attention_mask'])
            validation_token_type_ids.append(encoded_input['token_type_ids'])


train_input_ids = torch.tensor(train_input_ids)
aug_input_ids = torch.tensor(aug_input_ids)
validation_input_ids = torch.tensor(validation_input_ids)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_attention_masks = torch.tensor(train_attention_masks)
aug_attention_masks = torch.tensor(aug_attention_masks)
validation_attention_masks = torch.tensor(validation_attention_masks)

train_token_type_ids = torch.tensor(train_token_type_ids)
aug_token_type_ids = torch.tensor(aug_token_type_ids)
validation_token_type_ids = torch.tensor(validation_token_type_ids)


# Create the DataLoader
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels, train_token_type_ids,
                           aug_input_ids, aug_attention_masks, aug_token_type_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_input_ids, validation_attention_masks, validation_labels, validation_token_type_ids)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


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
                fps = fps + 1

    return tps, tns, fps, fns


def precision(tps, tns, fps, fns):
    return 100 * tps / max(tps + fps, 1)


def recall(tps, tns, fps, fns):
    return 100 * tps / max(tps + fns, 1)


def f1(precision, recall):
    return 2 * precision * recall / max(precision + recall, 1)


def train_model(model, linear_classifier, epochs=10):
    criterion = nn.CrossEntropyLoss()

    optimizer_bert = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    optimizer_classifier = AdamW(linear_classifier.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=total_steps)
    scheduler_classifier = get_linear_schedule_with_warmup(optimizer_classifier, num_warmup_steps=0, num_training_steps=total_steps)

    loss_values = []
    model.to(device)
    linear_classifier.to(device)
    best_f1 = 0

    for epoch_i in range(0, epochs):
        print_info = False
        if epoch_i % 1 == 0 or epoch_i == epochs-1:
            logging.info("")
            logging.info('======== Epoch {:} / {:} ========'.format(epoch_i, epochs-1))
            logging.info('Training...')
            print_info = True

        t0 = time.time()

        total_loss = 0
        model.train()
        linear_classifier.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_token_type_ids = batch[3].to(device)

            b_aug_input_ids = batch[4].to(device)
            b_aug_input_mask = batch[5].to(device)
            b_aug_token_type_ids = batch[6].to(device)

            model.zero_grad()
            linear_classifier.zero_grad()

            outputs_train = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)[1]
            outputs_aug = model(b_aug_input_ids, token_type_ids=b_aug_token_type_ids, attention_mask=b_aug_input_mask)[1]

            mixda_lambda = np.random.beta(0.9,0.1)
            # mixda_lambda = 0.9
            # mixda_lambda = np.random.beta(0.5,0.5)

            overall_outputs = torch.mul(outputs_train, mixda_lambda) + torch.mul(outputs_aug, (1-mixda_lambda))

            logits = linear_classifier(overall_outputs)

            loss = criterion(logits.view(-1, 2), b_labels.view(-1))

            total_loss += loss.item()

            if step % 100 == 0 and not step == 0 and print_info:
                elapsed = format_time(time.time() - t0)
                # Report progress.
                logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                logging.info("  Training Loss: {:}".format(loss))
                logging.info("  Average Training Loss: {:}".format(total_loss/(step+1)))

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer_bert.step()

            optimizer_classifier.step()

            scheduler_bert.step()

            scheduler_classifier.step()

        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)

        if print_info:
            logging.info("")
            logging.info("  Average training loss: {0:.5f}".format(avg_train_loss))
            logging.info("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            logging.info("Running Validation...")

        t0 = time.time()
        model.eval()
        linear_classifier.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        tps, tns, fps, fns = 0, 0, 0, 0

        for batch in validation_dataloader:
            batch = tuple(t for t in batch)
            b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)
            b_token_type_ids = b_token_type_ids.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
                logits = linear_classifier(outputs[1])

            # logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            tps, tns, fps, fns = update_stats(tps, tns, fps, fns, logits, label_ids)

        the_precision = precision(tps, tns, fps, fns)
        the_recall = recall(tps, tns, fps, fns)
        the_f1 = f1(the_precision, the_recall)

        if print_info:
            logging.info("Precision: {:} | Recall: {:} | F1: {:}".format(the_precision, the_recall, the_f1))
            logging.info("TPS: {:} | TNS: {:} | FPS: {:} | FNS: {:}".format(tps, tns, fps, fns))

            logging.info("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            logging.info("  Validation took: {:}".format(format_time(time.time() - t0)))

        print('Saving the model...')
        torch.save(model, os.path.join(MODEL_PATH, BERT_NAME))
        torch.save(linear_classifier, os.path.join(MODEL_PATH, CLASSIFIER_NAME))

        # if the_f1 >= best_f1:
        #     torch.save(model, os.path.join(MODEL_PATH, BERT_NAME+"_best"))
        #     torch.save(linear_classifier, os.path.join(MODEL_PATH, CLASSIFIER_NAME+'_best'))
        #     best_f1 = the_f1
        # print('Done.')

    logging.info("")
    logging.info("Training complete!")
    return loss_values


finttune_bert_loss_vals = train_model(model_finetune_bert, linear_classifier, epochs=epochs)