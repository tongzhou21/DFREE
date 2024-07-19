import json
import os
import random
random.seed(0)

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import tiktoken
import tqdm
import itertools
import math
import copy
from operator import itemgetter


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

def read_file(file_path):
    with open(file_path, 'r') as f_read:
        list_dict_data_labeled = [json.loads(line) for line in f_read]
    return list_dict_data_labeled



def balance_event(list_dict_data, total_data_count=10000):
    def resample_distribution(distribution, total=6000):
        def _resample_distribution(alpha, dis):
            dis_ratio = {key: math.log(value * alpha) for key, value in dis.items()}
            dis_count = {key: int(value / min(dis_ratio.values()) * min(dis.values()))
                         for key, value in dis_ratio.items()}
            return dis_count

        min_alpha = 1 / min(distribution.values())  # 0.006993006993006994
        dis_count = copy.deepcopy(distribution)
        max_alpha = 1e19
        try_count = 0

        while sum(dis_count.values()) != total:
            alpha = (min_alpha + max_alpha) / 2
            dis_count = _resample_distribution(alpha, distribution)
            if sum(dis_count.values()) < total:
                max_alpha = alpha
            else:
                min_alpha = alpha
            try_count += 1
            if try_count > 1e4:
                print('max try in balance', total_data_count)
                break
        if sum(dis_count.values()) > total:
            dict_category_cur = {key: value for key, value in distribution.items()}
            list_category_map = sum([[key for _ in range(value)] for key, value in dict_category_cur.items()], [])
            list_category_del = random.sample(list_category_map, sum(dis_count.values()) - total)
            for category_del in list_category_del:
                dis_count[category_del] -= 1

        elif sum(dis_count.values()) < total:
            dict_category_remain = {key: value - dis_count[key] for key, value in distribution.items()}
            list_category_map = sum([[key for _ in range(value)] for key, value in dict_category_remain.items()], [])
            list_category_add = random.sample(list_category_map, total - sum(dis_count.values()))
            for category_add in list_category_add:
                dis_count[category_add] += 1
        else:
            pass
        print('resample_distribution', sum(dis_count.values()))
        return dis_count

    dict_event_type2data_count = {key: 0 for key, _ in dict_event_type2arg_order.items()}
    for idx, dict_data in enumerate(list_dict_data):
        list_events = dict_data['events']
        list_event_type = list(set([d['event_type'] for d in list_events]))
        for event_type in list_event_type:
            dict_event_type2data_count[event_type] += 1
        dict_data['id'] = idx
    print(dict_event_type2data_count)


    def sort_dict_by_value_then_key(d):
        return dict(sorted(d.items(), key=itemgetter(1, 0)))


    dict_event_type2sample_count = resample_distribution(dict_event_type2data_count, total=total_data_count)


    dict_event_type2sample_count = sort_dict_by_value_then_key(dict_event_type2sample_count) # 从小到大排序
    print('dict_event_type2sample_count', dict_event_type2sample_count)

    dict_event2list_dict_data = {key: [] for key, _ in dict_event_type2arg_order.items()}
    for dict_data in list_dict_data:
        list_event_type = list(set([d['event_type'] for d in dict_data['events']]))
        for event_type in list_event_type:
            dict_event2list_dict_data[event_type].append(dict_data)

    list_dict_data_write = []
    set_id = set([])
    for event_type, sample_count in dict_event_type2sample_count.items():
        list_dict_data_cur = [dict_data for idx, dict_data in enumerate(dict_event2list_dict_data[event_type])
                              if dict_data['id'] not in set_id]
        list_dict_data_sampled= random.sample(list_dict_data_cur, min(sample_count, len(list_dict_data_cur)))
        list_dict_data_write += list_dict_data_sampled
        set_id |= set([d['id'] for d in list_dict_data_sampled])
    print('balance_event len(list_dict_data_write)', len(list_dict_data_write))
    random.shuffle(list_dict_data_write)
    return list_dict_data_write



def format_chatml(
        list_dict_data_labeled,
        file_path_write,
        prompt_template_file='prompt/prompt.template.v1.txt',
        max_length=8192,
):

    model_path = '/home/zhoutong/huggingface_model/Qwen/Qwen1.5-14B-Chat/'
    tokenizer_qwen = AutoTokenizer.from_pretrained(model_path)

    tokenizer_openai = tiktoken.get_encoding("cl100k_base")

    with open(prompt_template_file, 'r') as f_read:
        prompt_template = f_read.read()

    print('==> prompt_template')
    print(prompt_template)
    print('==> prompt_template')

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

    list_dict_data_write = []
    list_token_count_qwen = []
    list_token_count_openai = []
    error_count = 0
    random.shuffle(list_dict_data_labeled)
    for dict_data in tqdm.tqdm(list_dict_data_labeled, ncols=100, total=len(list_dict_data_labeled)):
        prompt = prompt_template.replace('{{TEXT}}', dict_data['content'])

        list_dict_event = dict_data['events']

        list_dict_event_ans = []
        for dict_event in list_dict_event:
            event_type = dict_event['event_type']
            dict_event_cur = {'event_type': event_type}
            for arg in dict_event_type2arg_order[event_type]:
                if arg in dict_event: dict_event_cur[arg] = dict_event[arg]
            list_dict_event_ans.append(dict_event_cur)
        str_dict_events = ',\n'.join([str(d) for d in list_dict_event_ans])
        output = f'[\n{str_dict_events}\n]'

        token_count = len(tokenizer_qwen.tokenize(prompt + output))
        if token_count > max_length:
            error_count += 1
            continue
        list_token_count_qwen.append(token_count)
        list_token_count_openai.append(len(tokenizer_openai.encode(prompt + output)))

        dict_data_write = {
            "type": "chatml",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": output,
                }
            ],
            "source": "unknown"
        }
        list_dict_data_write.append(dict_data_write)

    random.shuffle(list_dict_data_write)

    with open(file_path_write, 'w') as f_write:
        for dict_data_write in list_dict_data_write:
            f_write.write(json.dumps(dict_data_write, ensure_ascii=False) + '\n')

    print(np.percentile(np.array(list_token_count_qwen), [_ for _ in range(0, 105, 5)]))
    print(np.percentile(np.array(list_token_count_openai), [_ for _ in range(0, 105, 5)]))

    print('error_count', error_count, 'len(list_dict_data_write)', len(list_dict_data_write))



def find_all_positions(text, target):
    positions = []
    start = 0
    while True:
        start = text.find(target, start)
        if start == -1:
            break
        positions.append(start)
        start += 1
    return positions


def find_minimal_distance(positions):
    def max_distance(selection):
        if len(selection) < 2:
            return 0
        return max(abs(i - j) for i, j in itertools.combinations(selection, 2))

    keys = list(positions.keys())
    if not keys:
        return {}, 0

    all_combinations = itertools.product(*(positions[key] for key in keys))
    min_distance = float('inf')
    best_combination = None

    for combination in all_combinations:
        distance = max_distance(combination)
        if distance < min_distance:
            min_distance = distance
            best_combination = combination

    best_combination_dict = {key: value for key, value in zip(keys, best_combination)} if best_combination else {}

    return best_combination_dict, min_distance


def auto_tagging(list_dict_data):
    random.shuffle(list_dict_data)
    error_count = 0
    arg_count = 0
    event_count = 0

    list_arg_max_dis = []
    list_doc_length = []
    list_dis_ratio = []

    list_dict_data_write = []
    for dict_data in tqdm.tqdm(list_dict_data, ncols=100, total=len(list_dict_data)):
        text = dict_data['content']
        if len(text) == 0: continue

        for dict_event in dict_data['events']:
            event_count += 1
            dict_arg2positions = {}
            list_span = []
            for key, value in dict_event.items():
                if key == 'event_type' or key == 'event_id':
                    continue
                arg_count += 1
                if isinstance(value, str) == False or value == '':
                    error_count += 1
                    continue

                if value not in text:
                    error_count += 1
                else:
                    dict_arg2positions[key] = find_all_positions(text, value)
                    list_span.append(value)

            if len(dict_arg2positions.keys()) == 0: continue

            dict_arg2position, distance = find_minimal_distance(dict_arg2positions)

            list_arg_max_dis.append(distance)
            list_doc_length.append(len(text))
            list_dis_ratio.append(distance / len(text))

            dict_data_write = {
                'content': text,
                'event_type': dict_event['event_type'],
                'event_argument': [{
                    'start': start,
                    'end': start + len(dict_event[arg]),
                    'type': arg,
                    'text': dict_event[arg]
                } for arg, start in dict_arg2position.items()]
            }
            list_dict_data_write.append(dict_data_write)

    print(f"event_count: {event_count}, arg_count: {arg_count}, error_arg_count: {error_count}")

    return list_dict_data_write

def tag_result2mrc_data(list_dict_data_read):
    list_dict_data_write = []
    for dict_data in list_dict_data_read:
        context = dict_data['content']
        list_position = [d['start'] for d in dict_data['event_argument']] + \
                        [d['end'] for d in dict_data['event_argument']]

        span_offset_max = 20
        cur_text = context[min(0, min(list_position) - int(span_offset_max * random.random())):
                           max(list_position) + int(span_offset_max * random.random())]

        for dict_arg in dict_data['event_argument']:
            question = f"{dict_data['event_type']}事件中的{dict_arg['type']}"

            answer = {
                'text': dict_arg['text'],
                'answer_start': dict_arg['start'],
                'answer_end': dict_arg['end']
            }
            list_dict_data_write.append({
                'context': context,
                'context_target': cur_text,
                'question': question,
                'answer': answer,
            })
    return list_dict_data_write


def format_mrc(list_dict_data_labeled, file_path_write):
    list_dict_data_tagging = auto_tagging(
        list_dict_data=list_dict_data_labeled,
    )

    list_dict_data_write = tag_result2mrc_data(
        list_dict_data_read=list_dict_data_tagging,
    )

    with open(file_path_write, 'w') as f_write:
        for dict_data in list_dict_data_write:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

def format_ner(list_dict_data_labeled, file_path_write):
    list_dict_tagging = auto_tagging(list_dict_data_labeled)

    list_dict_data_write = []
    for dict_tagging in list_dict_tagging:
        list_tag = ['O' for _ in range(len(dict_tagging['content']))]
        for dict_arg in dict_tagging['event_argument']:
            list_tag[dict_arg['start']] = f"B-{dict_tagging['event_type']}-{dict_arg['type']}"
            for p in range(dict_arg['start'] + 1, dict_arg['end']):
                list_tag[p] = f"I-{dict_tagging['event_type']}-{dict_arg['type']}"

        list_dict_data_write.append({
            'content': dict_tagging['content'],
            'event_type': dict_tagging['event_type'],
            'tags': list_tag,
        })

    with open(file_path_write, 'w') as f_write:
        for dict_data in list_dict_data_write:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

def format_cls(list_dict_data_labeled, file_path_write):
    with open(file_path_write, 'w') as f_write:
        for dict_data in list_dict_data_labeled:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

    print(len(list_dict_data_labeled))


if __name__ == '__main__':
    #### cls
    format_cls(
        list_dict_data_labeled=read_file(
            'data/data_split/train.balance18k.jsonl',
        ),
        file_path_write='data/baseline_data/cls/cls.ee.train.balance18k.jsonl',
    )

    #### ner
    format_ner(
        list_dict_data_labeled = read_file(
            'data/data_split/train.balance18k.jsonl',
        ),
        file_path_write = 'data/baseline_data/ner/ner.ee.train.balance18k.jsonl',
    )


    #### mrc
    format_mrc(
        list_dict_data_labeled=read_file(
            'data/data_split/test.jsonl',
        ),
        file_path_write='data/baseline_data/mrc/mrc.ee.test.jsonl',
    )

    #### 准备sft数据
    format_chatml(
        list_dict_data_labeled=read_file(
            'data/data_split/train.balance18k.jsonl',
        ),
        file_path_write='data/baseline_data/sft/'
                        'chatml.ee.train.balance18k.prompt-v1-wodemo.tunc8192.jsonl',
        prompt_template_file='prompt/prompt.template.v1.wodemos.txt',
        max_length=8192,
    )


