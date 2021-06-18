from pyspark import SparkContext
from pyspark.mllib.fpm import FPGrowth
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


def write_output(FP_result, SON_result, output_path):
    FP_result2 = []
    for i in FP_result:
        if len(i) == 1:
            FP_result2.append(i[0])
        else:
            FP_result2.append(tuple(i))
    inter_len = len(list(set(FP_result2)&set(SON_result)))
    FP_len = len(FP_result)
    SON_len = len(SON_result)
    out_str = 'Task2,%d \nTask3,%d\nIntersection,%d' % (SON_len, FP_len, inter_len)
    with open(output_path, 'w') as f:
        f.write(out_str)


if __name__ == '__main__':
    '''
    k = 10
    support = 50
    input_path = 'task2_data.csv'
    output_path = 'task3_output'
    '''
    (k, support, input_path, output_path) = sys.argv[1:]
    start_time = time.time()
    bi_dat = csv_to_bi_data(input_path, 1)
    f_bi_dat = bi_dat.filter(lambda row: len(row) > k)
    total_basket = f_bi_dat.count()
    model = FPGrowth.train(f_bi_dat, support/total_basket)
    FPG_freq_set = model.freqItemsets().map(lambda row: row[0]).collect()
    print("Duration: %d s." % (time.time() - start_time))

    candidates = SON_stageI(f_bi_dat, int(support))
    SON_freq_set = SON_stageII(f_bi_dat, int(support), candidates)
    #print('SON result:', SON_freq_set)
    #print('FPG result:', FPG_freq_set)
    write_output(FPG_freq_set, SON_freq_set, output_path)
    #print('The FPG length:', len(FPG_freq_set))
    #print('The SON length:', len(SON_freq_set))
