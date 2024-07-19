import pickle
import json
import random
random.seed(0)
import os
import shutil
import copy
import torch
import os
import tqdm
from transformers import BertTokenizerFast, AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline
from utils import metric_api

def train(
        file_path_train='data/baseline_data/mrc/mrc.ee.labeled-4k.jsonl',
        file_path_dev = 'data/baseline_data/mrc/mrc.ee.dev.labeled.jsonl',
        MODEL_NAME="bert-base-chinese",
        chunk_size=448, overlap=64,
        epoch_count=2,
    ):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    folder_name_save = MODEL_NAME.replace('/', '-') + '.' + \
                       os.path.split(file_path_train)[-1].replace(".jsonl", '')
    if epoch_count != 2:
        folder_name_save = folder_name_save + f'.{epoch_count}epoch'
    if MODEL_NAME != 'bert-base-chinese':
        folder_name_save = folder_name_save + f".{MODEL_NAME.replace('/', '-')}"


    def read_data(file_path_read):
        print('read data from', file_path_read)

        with open(file_path_read, 'r') as f_read:
            list_dict_data = [json.loads(line) for line in f_read]
        random.shuffle(list_dict_data)

        train_contexts = [dict_data['context_target'] for dict_data in list_dict_data]
        train_questions = [dict_data['question'] for dict_data in list_dict_data]
        train_answers = [dict_data['answer'] for dict_data in list_dict_data]

        return train_contexts, train_questions, train_answers

    def read_data_chunk(file_path_read):
        print('read data from', file_path_read)
        with open(file_path_read, 'r') as f_read:
            list_dict_data = [json.loads(line) for line in f_read]
        random.shuffle(list_dict_data)

        list_dict_data_train = []
        for dict_data in list_dict_data:
            for idx_begin in range(0, len(dict_data['context']), chunk_size - overlap):
                chunk = dict_data['context'][idx_begin: idx_begin + chunk_size]
                if dict_data['answer']['text'] in chunk:
                    start = chunk.index(dict_data['answer']['text'])
                    end = start + len(dict_data['answer']['text'])
                    dict_data_train = {
                        'chunk': chunk,
                        'question': dict_data['question'],
                        'answer': {
                            'text': dict_data['answer']['text'],
                            'answer_start': start,
                            'answer_end': end,
                        }
                    }
                    list_dict_data_train.append(dict_data_train)

        train_contexts = [dict_data['chunk'] for dict_data in list_dict_data_train]
        train_questions = [dict_data['question'] for dict_data in list_dict_data_train]
        train_answers = [dict_data['answer'] for dict_data in list_dict_data_train]

        return train_contexts, train_questions, train_answers

    # train_contexts, train_questions, train_answers = read_data(file_path_train)
    # test_contexts, test_questions, test_answers = read_data(file_path_dev)
    train_contexts, train_questions, train_answers = read_data_chunk(file_path_train)
    test_contexts, test_questions, test_answers = read_data_chunk(file_path_dev)

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)


    def add_token_positions(encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    add_token_positions(train_encodings, train_answers)
    add_token_positions(test_encodings, test_answers)

    class SquadDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            try:
                return {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
            except KeyboardInterrupt:
                raise
            except:
                return self.__getitem__(idx + 1)

        def __len__(self):
            return len(self.encodings.input_ids)

    train_dataset = SquadDataset(train_encodings)
    test_dataset = SquadDataset(test_encodings)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)  # ("bert-base-uncased")

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=f"model_save/mrc/{folder_name_save}",  # output directory
        num_train_epochs=epoch_count,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        do_eval=True,
        logging_steps=100,
        learning_rate=2e-5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # save_only_model=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
    )
    print("all init done")
    trainer.train()


def predict(model_path):
    current_file_path = os.path.abspath(__file__)
    print('current_file_path', current_file_path)
    print('target', os.path.join(model_path, os.path.basename(current_file_path)))

    shutil.copyfile(current_file_path, model_path + 'code_predict.py')


    with open('model_save/cls/bert-base-chinese.cls.ee.train.m6a2.rand18k/checkpoint-14656/pred_res.jsonl', 'r') as f_read:
        list_dict_data_raw = [json.loads(line) for line in f_read]

    list_dict_data = []
    for dict_data in list_dict_data_raw:
        list_dict_data.append(dict_data)

    question_answerer = pipeline("question-answering", model=model_path, device=0)

    chunk_size, overlap = 512 - 64, 0
    min_score = 0.5
    doc_id2events_tgt, doc_id2events_pred = {}, {}
    for idx, dict_data in tqdm.tqdm(enumerate(list_dict_data), total=len(list_dict_data), ncols=100):
        text = dict_data['content']
        pred_events = dict_data['pred_label']

        list_chunk = [text[idx_begin: idx_begin + chunk_size]
                      for idx_begin in range(0, len(text), chunk_size - overlap)]

        dict_event_type2arg_order = {
            '破产清算': ['公司名称', '公司行业', '公告时间', '受理法院', '裁定时间'],
            '重大安全事故': ['伤亡人数', '公司名称', '公告时间', '其他影响', '损失金额'],
            '股东减持': ['减持开始日期', '减持的股东', '减持金额'],
            '股权质押': ['接收方', '质押开始日期', '质押方', '质押结束日期', '质押金额'],
            '股东增持': ['增持开始日期', '增持的股东', '增持金额'],
            '股权冻结': ['冻结开始日期', '冻结结束日期', '冻结金额', '被冻结股东'],
            '高层死亡': ['公司名称', '死亡/失联时间', '死亡年龄', '高层人员', '高层职务'],
            '重大资产损失': ['公司名称', '公告时间', '其他损失', '损失金额'],
            '重大对外赔付': ['公司名称', '公告时间', '赔付对象', '赔付金额']
        }

        list_arg_stream = []
        dict_pred = {'doc_id': idx, 'events': []}
        for event_name in pred_events:
            for arg_name in dict_event_type2arg_order[event_name]:
                for text_chunk in list_chunk:
                    question = f"{event_name}事件中的{arg_name}"
                    output = question_answerer(question=question, context=text_chunk)
                    if output['score'] > min_score:
                        list_arg_stream.append((event_name, arg_name, output['answer']))

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

    str_res = metric_api(doc_id2events_tgt, doc_id2events_pred)
    with open(model_path + 'eval_res.txt', 'w') as f_write:
        f_write.write(str_res)


if __name__ == "__main__":
    train(
        file_path_train='data/baseline_data/mrc/mrc.ee.train.rand18k.jsonl',
        file_path_dev='data/baseline_data/mrc/mrc.ee.test.jsonl',
        # MODEL_NAME="bert-base-chinese",
        MODEL_NAME="fnlp/bart-base-chinese",
        # MODEL_NAME="hfl/chinese-bert-wwm-ext",
        chunk_size=448,
        overlap=64,
        epoch_count=5
    )
    predict(
        model_path='model_save/mrc/bert-base-chinese.mrc.ee.train.m6a2.rand4k/checkpoint-6162/',
    )


'''
CUDA_VISIBLE_DEVICES=8 python baseline_bart_template.py


'''


