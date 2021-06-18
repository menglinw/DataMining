from pyspark import SparkContext
from itertools import combinations
from operator import add
import sys
import time


def csv_to_bi_data(input_path, case_num):
    '''
    read in csv data and ouput basket-item (bi) data, eliminate duplication within basket
    :return:
    basket-item data, corresponding to case_number required
    '''
    sc = SparkContext()
    text_dat = sc.textFile(input_path)
    header = text_dat.first()
    if int(case_num) not in [1, 2]:
        print('invalid case number, please check and re-enter')
    else:
        bi_dat = text_dat.filter(lambda row: row != header).map(lambda row: [row.split(',')[case_num-1],
                                                                             row.split(',')[case_num%2]]).\
            groupByKey().map(lambda row: sorted(list(set(row[1]))))
    return bi_dat


def A_Pri(iter, total_basket, support):
    '''
    A-Priori algorithm, input: partition iterator, output: local frequent itemsets
    :param iter: partition iterator
    :param total_basket: total number of basket
    :param support: total support threshold
    :return:
    a list of local frequent itemsets
    '''
    part = []
    for item in iter:
        part.append(item)
    adj_s = int(support * len(part) / int(total_basket))
    # find local frequent singleton
    item1_count = {}
    for basket in part:
        for item in basket:
            item1_count.setdefault(item, 0)
            item1_count[item] += 1
    freq_item_curr = sorted([key for key, value in item1_count.items() if value >= adj_s])
    freq_item = freq_item_curr
    item_size = 1

    # find larger local frequent itemsets
    while True:
        freq_item_prev = freq_item_curr
        if isinstance(freq_item_prev[0], tuple):
            temp_set = set()
            for tup in freq_item_prev:
                for ele in tup:
                    temp_set.add(ele)
            freq_item_prev = sorted([i for i in temp_set])
        item_size += 1
        item_count = {}
        for basket in part:
            for itemset in combinations(freq_item_prev, item_size):
                if set(itemset).issubset(basket):
                    item_count.setdefault(itemset, 0)
                    item_count[itemset] += 1
        freq_item_curr = [key for key, value in item_count.items() if value >= adj_s]
        if len(freq_item_curr) == 0:
            break
        freq_item += freq_item_curr
    return freq_item


def SON_stageI(bi_dat, support):
    '''
    Apply A-Priori algorithm to each partition and eliminate duplication of candidates
    :param bi_dat: basket-item data
    :param support: support threshold
    :return:
    all local frequent itemset candidates, without duplication
    '''
    total_basket = bi_dat.count()
    candidates = bi_dat.mapPartitions(lambda iter: A_Pri(iter, total_basket, support)).distinct().collect()
    return candidates


def count_all_candidates(iter, candidates):
    cand_count = {}
    for basket in iter:
        for itemset in candidates:
            if isinstance(itemset, str):
                if itemset in basket:
                    cand_count.setdefault(itemset, 0)
                    cand_count[itemset] += 1
            else:
                if set(itemset).issubset(basket):
                    cand_count.setdefault(tuple(sorted(itemset)), 0)
                    cand_count[tuple(sorted(itemset))] += 1
    return [(key, value) for key, value in cand_count.items()]


def SON_stageII(bi_dat, support, candidates):
    '''
    count all candidates in whole data and output the frequent itemsets
    :param bi_dat: basket-item data
    :param support: support threshold
    :param candidates: all frequent candidates
    :return:
    a list of frequent itemsets
    '''

    freq_set = bi_dat.mapPartitions(lambda iter: count_all_candidates(iter, candidates)).reduceByKey(add).\
        filter(lambda row: row[1] >= support).map(lambda row: row[0]).collect()
    return freq_set


def list_to_string(itemsets_list):
    # categorize itemsets by size, convert every element to string
    str_dict = {}
    for itemset in itemsets_list:
        if isinstance(itemset, str):
            str_dict.setdefault(0, [])
            str_dict[0].append(str(itemset))
        else:
            str_itemset = []
            for item in itemset:
                str_itemset.append(str(item))
            str_dict.setdefault(len(itemset)-1, [])
            str_dict[len(itemset)-1].append(tuple(str_itemset))
    # print(str_dict)
    # sort itemsets in the same size and format as output string
    output_str = ''
    for k in sorted(str_dict.keys()):
        str_dict[k] = sorted(str_dict[k])
        if k == 0:
            for i in range(len(str_dict[k])-1):
                output_str += '(\'' + str(str_dict[k][i]) + '\'),'
            output_str += '(\'' + str(str_dict[k][-1]) + '\') \n \n'
        else:
            for i in range(len(str_dict[k])-1):
                output_str += str(str_dict[k][i]) + ','
            output_str += str(str_dict[k][-1]) + '\n \n'
    # print(output_str)
    return output_str


def export_file(output_path, candidates, freq_set):
    with open(output_path, 'w') as output_file:
        writin_str = 'Candidates: \n' + list_to_string(candidates) +\
            'Frequent Itemsets:\n' + list_to_string(freq_set)
        output_file.write(writin_str)
        output_file.close()


if __name__ == '__main__':
    '''
    case_num = 1
    support = 7
    input_path = '../data/resource/asnlib/publicdata/small1.csv'
    output_path = 'task1_output'
    '''
    (case_num, support, input_path, output_path) = sys.argv[1:]
    start_time = time.time()
    bi_dat = csv_to_bi_data(input_path, int(case_num))
    #print('raw data: \n',bi_dat.collect())
    candidates = SON_stageI(bi_dat, int(support))
    freq_set = SON_stageII(bi_dat, int(support), candidates)
    #print('candidate: \n', candidates)
    #print('frequent: \n', freq_set)
    #print(list_to_string(candidates))
    export_file(str(output_path), candidates, freq_set)
    print("Duration: %d s." % (time.time() - start_time))


