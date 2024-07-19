import json
import os
import random

import tqdm
from zhconv import convert
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_token(text):
    token_ids = tokenizer.encode(text)
    return len(token_ids)

def read_notice(folder_path_read):
    list_file_name = os.listdir(folder_path_read)

    list_dict_data = []
    for idx, file_name in tqdm.tqdm(enumerate(list_file_name), ncols=100,
                                    total=len(list_file_name),
                                    desc=f'reading {folder_path_read} ...'):
        with open(folder_path_read + file_name, 'r') as f_read:
            list_dict_data += [json.loads(line) for line in f_read]

    return list_dict_data


def rule_category(dict_data):
    list_category = []

    text = dict_data['title'] + dict_data['content']
    dict_category2words = {
        '破产清算': ['破产'],
        '重大对外赔付': ['赔付', '赔偿', '索赔', '赔款'],
        '高层死亡':  ['去世', '逝世', '辞世'],
        '重大安全事故': ['事故'],
        '重大资产损失': ['损失', '亏损'],
        '股东减持': ['减持'],
        '股东增持': ['增持'],
        '股权冻结': ['冻结'],
        '股权质押': ['质押'],
    }
    text = convert(text, 'zh-hans')
    for category, list_word in dict_category2words.items():
        for word in list_word:
            if word in text:
                list_category.append(category)
                break

    dict_type2categories = {
        "股东/实际控制人股份减持": ["股东减持"],
        "破产清算": ["破产清算"],
        "股东/实际控制人股份增持": ["股东增持"],
        "股份质押、冻结": ["股权冻结", "股权质押"],
        "重大事故损失": ["重大资产损失", "重大安全事故"],
        "重大损失": ["重大资产损失"]
    }
    for type, categories in dict_type2categories.items():
        if type in dict_data['types']:
            list_category += categories

    if dict_data['title'] == '' or dict_data['content'] == '':
        list_category = []

    list_category = list(set(list_category))
    return list_category


def write_category_label(file_path_write='poi_data.all.jsonl'):
    list_folder_name = [f'result_20{y}/' for y in range(20, 24 + 1)]

    list_dict_data = []
    for folder_name in list_folder_name:
        list_dict_data += read_notice(folder_name)
    random.shuffle(list_dict_data)

    list_dict_data_hit = []
    list_dict_data_filter = []
    dict_category2count = {}
    for idx, dict_data in tqdm.tqdm(enumerate(list_dict_data), ncols=100,
                                    total=len(list_dict_data), desc='processing..'):

        list_category = rule_category(dict_data)

        if len(list_category) != 0:
            for category in list_category:
                if category not in dict_category2count:
                    dict_category2count[category] = 0
                dict_category2count[category] += 1
            dict_data['rule_category'] = list_category

            token_count = count_token(dict_data['content'])
            dict_data['token_count'] = token_count

            list_dict_data_hit.append(dict_data)
        else:
            list_dict_data_filter.append(dict_data)

    print('len(list_dict_data_hit), len(list_dict_data)', len(list_dict_data_hit), len(list_dict_data))

    with open(file_path_write, 'w') as f_write:
        for dict_data in list_dict_data_hit:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')
    print("ALL DONE, len(list_dict_data_hit), len(list_dict_data)", len(list_dict_data_hit), len(list_dict_data))


if __name__== '__main__':
    write_category_label(
        file_path_write='poi_data.all.jsonl'
    )

