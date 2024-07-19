import shutil
import copy
import json
import torch
import os
from transformers import BertTokenizerFast
import random
random.seed(0)
import tqdm
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import pipeline
from utils import metric_api

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, list_dict_data, tokenizer):
        self.list_dict_data = list_dict_data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(self.list_dict_data[idx]["tokens"],
                                          truncation=True,
                                          is_split_into_words=True)

        labels = []
        for i, label in enumerate([self.list_dict_data[idx][f"ner_tags"]]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels[0]

        return tokenized_inputs

    def __len__(self):
        return len(self.list_dict_data)

def train(file_path_train='data/baseline_data/ner/ner.ee.labeled-4k.jsonl',
          file_path_dev='data/baseline_data/ner/ner.ee.dev.labeled.jsonl',
          MODEL_NAME="bert-base-chinese", chunk_size=448, overlap=64,
          num_train_epochs=3,
          seed=42):


    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    folder_name_save = MODEL_NAME.replace('/', '-') + '.' + \
                       os.path.split(file_path_train)[-1].replace(".jsonl", '')
    if num_train_epochs != 3:
        folder_name_save = folder_name_save + f'.{num_train_epochs}epoch'
    if MODEL_NAME != "bert-base-chinese":
        folder_name_save += f".{MODEL_NAME.replace('/', '-')}"

    try:
        os.mkdir(f"model_save/ner/{folder_name_save}")
    except:
        pass

    current_file_path = os.path.abspath(__file__)
    print('current_file_path', current_file_path)
    shutil.copyfile(current_file_path,  f"model_save/ner/{folder_name_save}/code_train.py")


    print('file_path_train', file_path_train)
    print('folder_name_save', folder_name_save)

    with open(file_path_train, 'r') as f_read:
        list_dict_data = [json.loads(line) for line in f_read]

    id2label, label2id = {}, {}
    for dict_data in list_dict_data:
        labels = set(dict_data['tags'])
        for label in labels:
            if label not in label2id:
                id_cur = len(label2id)
                label2id[label] = id_cur
                id2label[id_cur] = label


    def get_chunked_data(file_path_train):
        with open(file_path_train, 'r') as f_read:
            list_dict_data = [json.loads(line) for line in f_read]

        list_dict_data_train = []
        drop_count = 0
        for idx, dict_data in enumerate(list_dict_data):
            ner_tags = [label2id[label] for label in dict_data['tags']]
            tokens = list(dict_data['content'])

            if len(ner_tags) != len(tokens):
                print("ERROR")

            for idx_begin in range(0, len(tokens), chunk_size - overlap):
                dict_data_train = {
                    'id': idx,
                    'ner_tags': ner_tags[idx_begin: idx_begin + chunk_size],
                    'tokens': tokens[idx_begin: idx_begin + chunk_size],
                }
                if min(dict_data_train['ner_tags']) == max(dict_data_train['ner_tags']): # 全是O
                    drop_count += 1
                    continue

                list_dict_data_train.append(dict_data_train)
        print('len(list_dict_data)', len(list_dict_data_train), 'drop_count', drop_count)
        return list_dict_data_train

    train_dataset = NERDataset(get_chunked_data(file_path_train), tokenizer)
    eval_dataset = NERDataset(get_chunked_data(file_path_dev), tokenizer)

    print('load dataset done')

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label.keys()),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"model_save/ner/{folder_name_save}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_steps=100,
        do_eval=True,
        evaluation_strategy='epoch',
        # save_strategy='no',
        save_strategy='epoch',
        # save_only_model=True,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    print("all init done")
    trainer.train()



def predict(model_path, test_path):
    current_file_path = os.path.abspath(__file__)
    print('current_file_path', current_file_path)
    print('target', os.path.join(model_path, os.path.basename(current_file_path)))
    shutil.copyfile(current_file_path, model_path + 'code_predict.py')

    #### 读取测试集合
    with open(test_path, 'r') as f_read:
        list_dict_data_raw = [json.loads(line) for line in f_read]

    print('len(list_dict_data_raw)',  len(list_dict_data_raw))

    list_dict_data = []
    for dict_data in list_dict_data_raw:
        for d_event in dict_data['events']:
            if 'event_type' not in d_event:
                print("ERROR")
                break
        else:
            list_dict_data.append(dict_data)

    classifier = pipeline("ner", model=model_path, device=0)

    event_strategy = 1
    print('event_strategy', event_strategy)
    chunk_size, overlap = 512 - 64, 0
    doc_id2events_tgt, doc_id2events_pred = {}, {}

    for idx, dict_data in tqdm.tqdm(enumerate(list_dict_data), total=len(list_dict_data), ncols=100):
        text = dict_data['content']
        list_chunk = [text[idx_begin: idx_begin + chunk_size]
                      for idx_begin in range(0, len(text), chunk_size - overlap)]

        if event_strategy == 0:
            dict_pred = {'doc_id': idx, 'events': []}

            for text_chunk in list_chunk:
                list_dict_label = classifier(text_chunk)

                prev_d_label = None
                cur_span = ''
                event_type2args = {}
                for d_label in list_dict_label:
                    if d_label['entity'][:1] == 'B':
                        if prev_d_label is not None and cur_span != '':
                            event_type = prev_d_label['entity'].split('-')[1]
                            arg_name = prev_d_label['entity'].split('-')[2]
                            span = cur_span
                            if event_type not in event_type2args:
                                event_type2args[event_type] = {'event_type': event_type}
                            event_type2args[event_type][arg_name] = span
                            cur_span = ''

                        cur_span = d_label['word']

                    elif d_label['entity'][:1] == 'I':
                        if prev_d_label is not None and \
                                prev_d_label['end'] == d_label['start'] and \
                                prev_d_label['entity'].replace('B', 'I') == d_label['entity'] and cur_span != '':
                            cur_span += d_label['word']
                        else:
                            if prev_d_label is not None and cur_span != '':
                                event_type = prev_d_label['entity'].split('-')[1]
                                arg_name = prev_d_label['entity'].split('-')[2]
                                span = cur_span
                                if event_type not in event_type2args:
                                    event_type2args[event_type] = {'event_type': event_type}
                                event_type2args[event_type][arg_name] = span
                                cur_span = ''

                    prev_d_label = copy.deepcopy(d_label)
                else:
                    if prev_d_label is not None and cur_span != '':
                        event_type = prev_d_label['entity'].split('-')[1]
                        arg_name = prev_d_label['entity'].split('-')[2]
                        span = cur_span
                        if event_type not in event_type2args:
                            event_type2args[event_type] = {'event_type': event_type}
                        event_type2args[event_type][arg_name] = span
                        cur_span = ''
                list_d_args = [d_args for _, d_args in event_type2args.items()]
                if len(list_d_args) != 0:
                    dict_pred['events'] += list_d_args

            doc_id2events_pred[idx] = copy.deepcopy(dict_pred['events'])
            doc_id2events_tgt[idx] = copy.deepcopy(dict_data['events'])
        elif event_strategy == 1:
            dict_pred = {'doc_id': idx, 'events': []}
            list_arg_stream = []
            for text_chunk in list_chunk:
                list_dict_label = classifier(text_chunk)

                prev_d_label = None
                cur_span = ''
                for d_label in list_dict_label:
                    if d_label['entity'][:1] == 'B':
                        if prev_d_label is not None and cur_span != '':
                            event_type = prev_d_label['entity'].split('-')[1]
                            arg_name = prev_d_label['entity'].split('-')[2]
                            span = cur_span
                            list_arg_stream.append((event_type, arg_name, span))
                            cur_span = ''

                        cur_span = d_label['word']

                    elif d_label['entity'][:1] == 'I':
                        if prev_d_label is not None and \
                                prev_d_label['end'] == d_label['start'] and \
                                prev_d_label['entity'].replace('B', 'I') == d_label['entity'] and cur_span != '':
                            cur_span += d_label['word']
                        else:
                            if prev_d_label is not None and cur_span != '':
                                event_type = prev_d_label['entity'].split('-')[1]
                                arg_name = prev_d_label['entity'].split('-')[2]
                                span = cur_span
                                list_arg_stream.append((event_type, arg_name, span))
                                cur_span = ''

                    prev_d_label = copy.deepcopy(d_label)
                else:
                    if prev_d_label is not None and cur_span != '':
                        event_type = prev_d_label['entity'].split('-')[1]
                        arg_name = prev_d_label['entity'].split('-')[2]
                        span = cur_span
                        list_arg_stream.append((event_type, arg_name, span))
                        cur_span = ''


            event_type2list_event = {}
            for tuple_arg in list_arg_stream:
                if tuple_arg[0] not in event_type2list_event:
                    event_type2list_event[tuple_arg[0]] = [{'event_type': tuple_arg[0]}]

                if tuple_arg[1] not in event_type2list_event[tuple_arg[0]][-1]:
                    event_type2list_event[tuple_arg[0]][-1][tuple_arg[1]] = tuple_arg[2]
                else:
                    if event_type2list_event[tuple_arg[0]][-1][tuple_arg[1]] == tuple_arg[2]:
                        continue
                    else:
                        event_type2list_event[tuple_arg[0]].append({'event_type': tuple_arg[0]})
                        event_type2list_event[tuple_arg[0]][-1][tuple_arg[1]] = tuple_arg[2]

            for _, list_event in event_type2list_event.items():
                if len(list_event) != 0:
                    dict_pred['events'] += list_event

            doc_id2events_pred[idx] = copy.deepcopy(dict_pred['events'])
            doc_id2events_tgt[idx] = copy.deepcopy(dict_data['events'])
        else:
            print("ERROR in event_strategy", event_strategy)
            raise

    str_res = metric_api(doc_id2events_tgt, doc_id2events_pred)
    with open(model_path + 'eval_res.txt', 'w') as f_write:
        f_write.write(str_res)






if __name__ == '__main__':
    train(
        file_path_train='data/baseline_data/ner/ner.ee.train.rand18k.jsonl',
        file_path_dev='data/baseline_data/ner/ner.ee.test.jsonl',
        MODEL_NAME="bert-base-chinese",
        # MODEL_NAME="hfl/chinese-bert-wwm-ext",
        chunk_size=448, overlap=64,
        num_train_epochs=10,
        seed=42
    )
    predict(
        model_path="model_save/ner/bert-base-chinese.ner.ee.train.rand18k/checkpoint-36520/",
        test_path='data/data_split/test.jsonl'
    )




'''
CUDA_VISIBLE_DEVICES=4 python baseline_bert_tagging.py

'''






