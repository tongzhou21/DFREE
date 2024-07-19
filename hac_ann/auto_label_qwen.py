import json
import os
import tqdm
from openai import OpenAI
import tiktoken
import re
import time

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


def get_response_qwen_vllm(question, url=None):
    openai_api_key = "EMPTY"
    if not url:
        openai_api_base = "http://localhost:10032/v1"
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


def fin_ie(file_path_read, folder_path_write, url, try_count=3):
    file_path_write = folder_path_write + os.path.split(file_path_read)[1].replace('.jsonl', '.out.jsonl')

    with open(file_path_write, 'w') as f_write: pass

    with open('prompt/prompt.template.v1.txt', 'r') as f_read:
        prompt_template = f_read.read()

    with open(file_path_read, 'r') as f_read:
        for idx, line in tqdm.tqdm(enumerate(f_read), ncols=100, total=10000):
            dict_data = json.loads(line)
            text = normalize_text(dict_data['content'])
            text = text.replace('\n', '<br>').replace('<br><br>', '<br>')

            prompt = prompt_template.replace('{{TEXT}}', text)

            for _ in range(try_count):
                try:
                    output = get_response_qwen_vllm(prompt, url)

                    dict_data['text_in'] = text
                    dict_data['output'] = output
                    if format_ans(output) is not None:
                        break
                except KeyboardInterrupt:
                    raise
                except:
                    print("ERROR", idx)
                    time.sleep(2)
            with open(file_path_write, 'a') as f_write:
                f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

    print("ALL DONE")


if __name__ == '__main__':
    output = get_response_qwen_vllm('hello', 'http://localhost:11032/v1')
    print(output)

    fin_ie(
        file_path_read='data/auto_label/inp/poi_data.1of38.jsonl',
        folder_path_write='data/auto_label/out_qwen_1.5_32B/poi_data.1of38.jsonl',
        url='http://localhost:11032/v1',
        try_count=3
    )


'''
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
    --served-model-name Qwen1.5-32B-Chat \
    --model Qwen/Qwen1.5-32B-Chat \
    --max-model-len 16384 \
    --port 11032
'''



