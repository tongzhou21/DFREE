import shutil
import json
import torch
import os
from transformers import BertTokenizerFast
import random

random.seed(0)
import tqdm
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, list_dict_data, tokenizer):
        self.list_dict_data = list_dict_data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(self.list_dict_data[idx]["chunk"],
                                          padding=True,
                                          truncation=True)

        tokenized_inputs["labels"] = [float(v) for v in self.list_dict_data[idx]['labels']]

        return tokenized_inputs

    def __len__(self):
        return len(self.list_dict_data)


def train(file_path_train='data/baseline_data/cls/cls.ee.labeled-4k.jsonl',
          file_path_dev='data/baseline_data/cls/cls.ee.dev.labeled.jsonl',
          MODEL_NAME="bert-base-chinese", chunk_size=448, overlap=64,
          num_train_epochs=3,
          seed=42):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    folder_name_save = MODEL_NAME.replace('/', '-') + '.' + \
                       os.path.split(file_path_train)[-1].replace(".jsonl", '')

    current_file_path = os.path.abspath(__file__)
    print('current_file_path', current_file_path)
    try:
        os.mkdir(f"model_save/cls/{folder_name_save}")
        shutil.copyfile(current_file_path, f"model_save/cls/{folder_name_save}/code_train.py")
    except: pass

    print('file_path_train', file_path_train)
    print('folder_name_save', folder_name_save)

    with open(file_path_train, 'r') as f_read:
        list_dict_data = [json.loads(line) for line in f_read]

    id2label, label2id = {}, {}
    for dict_data in list_dict_data:
        labels = set([d['event_type'] for d in dict_data['events']])
        for label in labels:
            if label not in label2id:
                id_cur = len(label2id)
                label2id[label] = id_cur
                id2label[id_cur] = label
    label_count = len(label2id.keys())
    print(len(label2id.keys()), label2id)

    def get_chunked_data(file_path_train):
        with open(file_path_train, 'r') as f_read:
            list_dict_data = [json.loads(line) for line in f_read]

        list_dict_data_train = []
        for idx, dict_data in enumerate(list_dict_data):
            list_label_name = list(set([d['event_type'] for d in dict_data['events']]))
            labels = [0 for _ in range(label_count)]
            for label_name in list_label_name:
                labels[label2id[label_name]] = 1

            for idx_begin in range(0, len(dict_data['content']), chunk_size - overlap):
                dict_data_train = {
                    'id': idx,
                    'chunk': dict_data['content'][idx_begin: idx_begin + chunk_size],
                    'labels': labels,
                }

                list_dict_data_train.append(dict_data_train)
        print('len(list_dict_data)', len(list_dict_data_train))
        return list_dict_data_train

    train_dataset = ClassificationDataset(get_chunked_data(file_path_train), tokenizer)
    eval_dataset = ClassificationDataset(get_chunked_data(file_path_dev), tokenizer)

    print('load dataset done')

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        problem_type="multi_label_classification",
        num_labels=label_count,
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"model_save/cls/{folder_name_save}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_steps=100,
        do_eval=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
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

    print('len(list_dict_data_raw)', len(list_dict_data_raw))

    list_dict_data = []
    for dict_data in list_dict_data_raw:
        for d_event in dict_data['events']:
            if 'event_type' not in d_event:
                print("ERROR")
                break
        else:
            list_dict_data.append(dict_data)

    classifier = pipeline("text-classification", model=model_path, device=0)

    chunk_size, overlap = 512 - 64, 0
    correct, pred, tgt = 0, 1e-9, 1e-9

    list_dict_data_write = []
    for idx, dict_data in tqdm.tqdm(enumerate(list_dict_data), total=len(list_dict_data), ncols=100):
        text = dict_data['content']
        target_label = [d['event_type'] for d in dict_data['events']]

        list_chunk = [text[idx_begin: idx_begin + chunk_size]
                      for idx_begin in range(0, len(text), chunk_size - overlap)]
        pred_label = []
        for text_chunk in list_chunk:
            list_dict_label = classifier(text_chunk)
            pred_label += [d['label'] for d in list_dict_label]


        correct += len(set(pred_label) & set(target_label))
        pred += len(set(pred_label))
        tgt += len(set(target_label))

        dict_data['pred_label'] = list(set(pred_label))
        list_dict_data_write.append(dict_data)

    with open(model_path + 'pred_res.jsonl', 'w') as f_write:
        for dict_data in list_dict_data_write:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')


    str_res = f'event_type ==> P: {correct / pred:.4f}, R: {correct / tgt:.4f}, ' \
              f'F: {2 * ((correct / pred) * (correct / tgt)) / ((correct / pred) + (correct / tgt) + 1e-9):.4f}'
    print(str_res)
    with open(model_path + 'eval_res.txt', 'w') as f_write:
        f_write.write(str_res)


if __name__ == '__main__':
    train(
        file_path_train='data/baseline_data/cls/cls.ee.train.rand18k.jsonl',
        file_path_dev='data/baseline_data/cls/cls.ee.test.jsonl',
        MODEL_NAME="bert-base-chinese", chunk_size=448, overlap=64,
        num_train_epochs=2,
        seed=42
    )
    predict(
        model_path='model_save/cls/bert-base-chinese.cls.ee.train.rand18k/checkpoint-14656/',
        test_path='data/baseline_data/cls/cls.ee.test.jsonl'
    )

'''

CUDA_VISIBLE_DEVICES=4 python baseline_bert_classification.py

'''






