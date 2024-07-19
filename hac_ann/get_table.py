import re

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




def auto_retry(dict_data, str_output):
    list_dict_event = format_ans(str_output)

    if list_dict_event is None:
        return True

    list_event_type = [d['event_type'] for d in list_dict_event]
    if len(set(list_event_type) - set(dict_data['rule_category'])) != 0:
        return True

    for dict_event in list_dict_event:
        for key, value in dict_event.items():
            if key != 'event_type' and key not in dict_event_type2arg_order[dict_event['event_type']]:
                return True

    def remove_whitespace(input_string):
        try:
            res = re.sub(r'\s+', '', input_string)
        except:
            print('ERROR', input_string)
            raise
        return res

    for dict_event in list_dict_event:
        for key, value in dict_event.items():
            if isinstance(value, str) == False:
                return True

            if key != 'event_type' and remove_whitespace(value) not in remove_whitespace(dict_data['content']):
                return True

    return False


if __name__ == '__main__':
    pass

