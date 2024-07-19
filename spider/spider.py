import json
import time
import requests
from datetime import datetime, timedelta
import random


def get_response(url, request_interval=0.25):
    time.sleep((request_interval / 1.5) * (1 + random.random()))
    response = requests.get(url)
    return response

def spider_date(datetime='2024-04-01'):
    list_notice = []
    for page_index in range(1, 200):
        url = f"https://np-anotice-stock.eastmoney.com/api/security/ann?cb=jQuery11230955024237544617_1712809299240&sr=-1" \
              f"&page_size=50&page_index={page_index}&ann_type=SHA%2CCYB%2CSZA%2CBJA%2CINV&client_source=web&f_node=0&s_node=0&" \
              f"begin_time={datetime}&end_time={datetime}"

        response = get_response(url)

        json_str = response.text[response.text.find("(") + 1: response.text.rfind(")")]
        dict_data = json.loads(json_str)
        for idx, dict_info in enumerate(dict_data['data']['list']):
            list_notice.append({
                'art_code': dict_info['art_code'],
                'title': dict_info['title'],
                'company': dict_info['codes'][0]['short_name'],
                'code': dict_info['codes'][0]['stock_code'],
                'types': [d['column_name'] for d in dict_info['columns']]
            })

        if len(dict_data['data']['list']) < 50:
            break
    return list_notice


def spider_notice(notice_id='AN202404011629212556'):
    dict_res = {
        'art_code': notice_id,
        'pdf': '',
        'title': '',
        'content': '',
    }
    for page_index in range(1, 10):
        url = f"https://np-cnotice-stock.eastmoney.com/api/content/ann?cb=jQuery112305882817409394032_1712810905720" \
              f"&art_code={notice_id}&client_source=web&page_index={page_index}&_=1712810905721"

        response = get_response(url)

        json_str = response.text[response.text.find("(") + 1: response.text.rfind(")")]
        dict_data = json.loads(json_str)

        dict_res['art_code'] = dict_data['data']['art_code']
        dict_res['pdf'] = dict_data['data']['attach_url']
        dict_res['title'] = dict_data['data']['notice_title']
        dict_res['content'] += dict_data['data']['notice_content'] + '\n'

        page_size = dict_data['data']['page_size']
        if page_index == int(page_size):
            break

    return dict_res


def spider():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 4, 10)

    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print('date_str', date_str, end='\r')

        list_dict_notice = spider_date(datetime=date_str)
        print('notice_count', len(list_dict_notice), end='\r')

        if len(list_dict_notice) == 0: continue

        file_path_write = f'result/notice.{date_str}.jsonl'
        with open(file_path_write, 'w') as f_write: pass

        for idx, dict_notice in enumerate(list_dict_notice):
            dict_res = spider_notice(dict_notice['art_code'])
            dict_res['code'] = dict_notice['code']
            dict_res['company'] = dict_notice['company']
            dict_res['types'] = dict_notice['types']

            with open(file_path_write, 'a') as f_write:
                f_write.write(json.dumps(dict_res, ensure_ascii=False) + '\n')
            print(f'{date_str}: ({idx + 1} / {len(list_dict_notice)}) notices', end='\r')
        print()
        current_date += timedelta(days=1)
    print('ALL DONE')

if __name__== '__main__':
    spider()



