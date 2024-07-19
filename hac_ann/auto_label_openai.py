import json
import os
import random
random.seed(0)

import tqdm
from openai import OpenAI
import tiktoken
import re
import time
from get_table import auto_retry

tokenizer = tiktoken.get_encoding("cl100k_base")
def token_count(text):
    token_ids = tokenizer.encode(text)
    return len(token_ids)


def format_ans(str_output):
    try:
        pattern = r'\[.*\]'
        match = re.search(pattern, str_output.replace('\n', ''))

        if match:
            list_dict_res = eval(match.group(0))
        else:
            list_dict_res = eval(f"[{str_output}]")

        list_event_type = ['破产清算', '重大安全事故', '股东减持', '股权质押',
                           '股东增持', '股权冻结', '高层死亡', '重大资产损失', '重大对外赔付']
        for dict_event in list_dict_res:
            if dict_event['event_type'] not in list_event_type:
                list_dict_res = None
                break
    except KeyboardInterrupt:
        raise
    except Exception as e:
        list_dict_res = None

    return list_dict_res

def normalize_text(text):
    text = text.strip()
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.replace('\t', ' ')

    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)

    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.strip()
    return text


def get_response_openai(question, model='gpt-3.5-turbo-0125', temperature=0.6):
    time_begin = time.time()
    assert model in ['gpt-3.5-turbo-0125']
    my_key = ''
    client = OpenAI(api_key=my_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        print(f'get_response_openai successed in {time.time() - time_begin:.1f}s')
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f'get_response_openai failed in {time.time() - time_begin:.1f}s')
        return str(e)


def get_response_qwen_vllm(question, model=None, url=None, temperature=None):
    openai_api_key = "EMPTY"
    if not url:
        openai_api_base = "http://localhost:10032/v1" #
    else:
        openai_api_base = url

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen1.5-32B-Chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
    )
    output = chat_response.choices[0].message.content.strip()
    return output




def fin_ie_v3(file_path_write, prompt_path, model_name, max_try_count=4):
    with open(file_path_write, 'w') as f_write:
        pass

    with open(prompt_path, 'r') as f_read:
        prompt_template = f_read.read()

    folder_path_read = 'data/auto_label/out_stage2_gpt_3.5/'

    list_file_path_result = [folder_path_read + file_name
                             for file_name in os.listdir(folder_path_read) if 'v2' in file_name]
    list_dict_data = []
    for file_path_result in list_file_path_result:
        with open(file_path_result, 'r') as f_read:
            list_dict_data += [json.loads(line) for line in f_read]
    random.shuffle(list_dict_data)

    list_dict_data_remain = []
    for dict_data in list_dict_data:
        str_output = dict_data['output_openai_0']
        if 'output_openai_1' in dict_data:
            str_output = dict_data['output_openai_1']

        if auto_retry(dict_data, str_output) is True:
            list_dict_data_remain.append(dict_data)

    random.shuffle(list_dict_data_remain)

    request_count, success_count = 0, 0
    for idx, dict_data in tqdm.tqdm(enumerate(list_dict_data_remain), ncols=100, total=len(list_dict_data_remain)):
        text = dict_data['text_in']

        prompt = prompt_template.replace('{{TEXT}}', text)

        dict_data['output_openai_retry'] = []
        for idx_try in range(max_try_count):
            try:
                output = get_response_openai(prompt, temperature=0.5)
                # output = get_response_qwen_vllm(prompt) # TODO
                request_count += 1

                dict_data['output_openai_retry'].append(output)

                if auto_retry(dict_data, output):
                    continue
                else:
                    success_count += 1
                    break
            except KeyboardInterrupt:
                raise
            except:
                print("ERROR", idx)
                time.sleep(2)
        else:
            print(f'{idx} failed in {max_try_count} retries')

        print(f'** request: {request_count}, success: {success_count} **')

        with open(file_path_write, 'a') as f_write:
            f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

    print("ALL DONE")

if __name__ == '__main__':
    output = get_response_openai('hello', model='gpt-3.5-turbo-0125')
    print(output)

    fin_ie_v3(
        file_path_write=f'data/auto_label/out_stage3_gpt_3.5/'
                        f'poi_data.stage3.retry.out.jsonl',
        prompt_path='prompt/prompt.template.v2.txt',
        model_name='gpt-3.5-turbo-0125',
        max_try_count=5
    )


