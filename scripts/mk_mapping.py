# this code is writen by Alexander Firsov
# https://www.kaggle.com/alexfir/mapping-between-test-supplement-csv-and-test-csv

old_file_path = '../input/test_supplement.csv'
file_path = '../input/test.csv'
output_file_path = '../input/mapping.csv'


def _split(line):
    line = line.strip()
    index = line.index(',')
    last_index = line.rindex(',')
    click_id = line[:index]
    payload = line[index:]
    time = line[last_index:]
    return click_id, payload, time

def _read_same_time(lines, unprocessed_line):
    click_id, payload, group_time = _split(unprocessed_line)
    click_id_dict = {payload: [click_id]}
    while True:
        unprocessed_line = lines.readline()
        if not unprocessed_line:
            return unprocessed_line, click_id_dict, group_time
        click_id, payload, click_time = _split(unprocessed_line)
        if group_time == click_time:
            if payload in click_id_dict:
                click_id_dict[payload].append(click_id)
            else:
                click_id_dict[payload] = [click_id]
        else:
            return unprocessed_line, click_id_dict, group_time

def _find_time(lines, group_time, unprocessed_line):
    if unprocessed_line:
        click_id, payload, time = _split(unprocessed_line)
        if group_time == time:
            return unprocessed_line
    while True:
        unprocessed_line = lines.readline()
        click_id, payload, time = _split(unprocessed_line)
        if group_time == time:
            return unprocessed_line


def _save(output, test_click_id_dict, old_test_click_id_dict):
    for payload, click_ids in test_click_id_dict.items():
        old_click_ids = old_test_click_id_dict[payload]
        if len(old_click_ids) != len(click_ids):
            print('Number of ids mismatch for "{}", test ids = {}, old test ids = {}'.format(payload, click_ids,
                                                                                             old_click_ids))
        for i in range(len(click_ids)):
            output.write('{},{}\n'.format(click_ids[i], old_click_ids[i]))


with open(file_path, "r", encoding="utf-8") as test:
    with open(old_file_path, "r", encoding="utf-8") as old_test:
        with open(output_file_path, "w", encoding="utf-8") as output:
            output.write('click_id,old_click_id\n')
            test.readline()  # skip header
            old_test.readline()  # skip header
            old_test_unprocessed_line = old_test.readline()
            test_unprocessed_line = test.readline()
            while test_unprocessed_line != '':
                test_unprocessed_line, test_click_id_dict, click_time = _read_same_time(test, test_unprocessed_line)
                old_test_unprocessed_line = _find_time(old_test, click_time, old_test_unprocessed_line)
                old_test_unprocessed_line, old_test_click_id_dict, _ = _read_same_time(old_test,
                                                                                       old_test_unprocessed_line)
                _save(output, test_click_id_dict, old_test_click_id_dict)
        pass


import pandas as pd
mapping = pd.read_csv(output_file_path, dtype={'click_id': 'int32','old_click_id': 'int32'}, engine='c',
                na_filter=False,memory_map=True)

