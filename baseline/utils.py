# coding=utf8
import copy
import json

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

class EventTypes:
    ALL_EVENTS = ['破产清算',
                  '重大安全事故',
                  '股东减持',
                  '股权质押',
                  '股东增持',
                  "股权冻结",
                  "高层死亡",
                  "重大资产损失",
                  "重大对外赔付"]
    NUM_EVENT_TYPES = len(ALL_EVENTS)


def find_the_most_similar(roles_list_label,roles):
    most_similar_roles = None
    biggest_num_hit = 0
    for roles_label in roles_list_label:
        num_hit = len(set(roles_label).intersection(set(roles)))
        if num_hit>biggest_num_hit:
            biggest_num_hit = num_hit
            most_similar_roles = roles_label
    return biggest_num_hit,most_similar_roles


def convert_to_dict(events):
    new_events = dict()
    for event in events:
        if "event_id" in event:
            del event["event_id"]
        event_type = event["event_type"]

        roles_list = []
        for role,entity in event.items():
            if role != 'event_type':
                roles_list.append(role+":"+entity)
        if event_type in new_events:
            new_events[event_type].append(roles_list)
        else:
            new_events[event_type] = [roles_list]
    return new_events


def evaluate_single_sample(label,pred):
    label = convert_to_dict(label)
    pred = convert_to_dict(pred)
    num_total = {event:sum(map(lambda x:len(x),roles_list))
                for event,roles_list in label.items()}

    num_pred = {event:sum(map(lambda x:len(x),roles_list))
                for event,roles_list in pred.items()}

    num_hit = dict()
    for event in EventTypes.ALL_EVENTS:
        if event not in num_total:
            num_total[event] = 0
        if event not in num_pred:
            num_pred[event] = 0
        num_hit[event] = 0

    for event,roles_list in label.items():
        if event not in pred:
            continue
        for roles in roles_list:
            hit,most_similar_roles = find_the_most_similar(pred[event],roles)
            if most_similar_roles:
                pred[event].remove(most_similar_roles)
            num_hit[event] += hit

    return num_total,num_pred,num_hit


def evaluate(labels,preds):
    n_groundtruth = dict(zip(EventTypes.ALL_EVENTS,[0]*EventTypes.NUM_EVENT_TYPES))
    n_pred = dict(zip(EventTypes.ALL_EVENTS,[0]*EventTypes.NUM_EVENT_TYPES))
    n_hit = dict(zip(EventTypes.ALL_EVENTS,[0]*EventTypes.NUM_EVENT_TYPES))
    for docid in labels.keys():
        label = labels[docid]
        pred = preds[docid] if docid in preds else []
        num_groundtruth,num_pred,num_hit = evaluate_single_sample(label,pred)
        for event in num_groundtruth.keys():
            n_groundtruth[event] += num_groundtruth[event]
            n_pred[event] += num_pred[event]
            n_hit[event] += num_hit[event]

    event_precision = dict()
    event_recall = dict()
    event_f1 = dict()
    for event in EventTypes.ALL_EVENTS:
        precision = float(n_hit[event]) / n_pred[event] if n_pred[event] else 0.0
        recall = float(n_hit[event]) / n_groundtruth[event] if n_groundtruth[event] else 0.0
        event_precision[event] = precision
        event_recall[event] = recall
        event_f1[event] = (2*precision*recall)/(recall+precision) if (recall+precision) else 0.0

    n_total_pred = sum([n_pred[event] for event in EventTypes.ALL_EVENTS])
    n_total_hit = sum([n_hit[event] for event in EventTypes.ALL_EVENTS])
    n_total_gt = sum([n_groundtruth[event] for event in EventTypes.ALL_EVENTS])

    total_P = float(n_total_hit)/n_total_pred if n_total_pred else 0.0
    total_R = float(n_total_hit)/n_total_gt if n_total_gt else 0.0
    total_f1 = (2*total_P*total_R)/(total_P+total_R) if (total_R+total_P) else 0.0
    return total_P,total_R,total_f1,event_precision,event_recall,event_f1


def __metric_api(label_file,pred_file):
    labels = dict()
    for line in open(label_file, 'r').readlines():
        line = json.loads(line)
        labels[line["doc_id"]] = line["events"]

    preds = dict()
    for line in open(pred_file, 'r').readlines():
        line = json.loads(line)
        preds[line["doc_id"]] = line["events"]

    total_P, total_R, total_f1, event_precision, event_recall, event_f1 = evaluate(labels, preds)

    print("Overall P:%f,R:%f,F1:%f" % (total_P, total_R, total_f1))
    for event in event_precision.keys():
        print("%s,P:%f,R:%f,F1:%f" % (event, event_precision[event], event_recall[event], event_f1[event]))

    return total_P, total_R, total_f1


def metric_api(doc_id2events_tgt, doc_id2events_pred):
    for id, events in doc_id2events_tgt.items():
        events_new = [{key: value for key, value in d.items() if (key != 'event_id' and value != '')} for d in events]
        doc_id2events_tgt[id] = copy.deepcopy(events_new)

    for id, events in doc_id2events_pred.items():
        events_new = [{key: value for key, value in d.items() if (key != 'event_id' and value != '')} for d in events]
        doc_id2events_pred[id] = copy.deepcopy(events_new)

    total_P, total_R, total_f1, event_precision, event_recall, event_f1 = evaluate(doc_id2events_tgt, doc_id2events_pred)


    str_res = f"Overall ====> P: {total_P:.4f}, R: {total_R:.4f}, F1: {total_f1:.4f}"

    for event in event_precision.keys():
        str_res += '\n' + f"-> {event} => P: {event_precision[event]:.4f}, R: {event_recall[event]:.4f}, F1: {event_f1[event]:.4f}"
    return str_res
