from pyspark import SparkContext
import json
import sys
import random
import math
import time
import itertools


def perm_hash(row, m, num_perm, user_index):
    '''
    take a row [b_id, u_id] as input and output a list of hashed result of u_id
    :param row: [b_id, u_id]
    :param m: number of basket
    :param num_perm: number of permutation hash function
    :return: [b_id, h1, h2, h3...]
    '''
    num_perm = int(num_perm)
    m = int(m)
    list_hash = []
    list_a = random.sample(list(range(num_perm*50)), num_perm)
    list_b = random.sample(list(range(num_perm*50)), num_perm)

    def gen_hash(a, b, m):
        def out_hash(x):
            return (x*a + b)%m
        return out_hash
    for a, b in zip(list_a, list_b):
        list_hash.append(gen_hash(a,b,m))
    return [user_index[row], [p_hash(user_index[row]) for p_hash in list_hash]]


def min_list(list_a, list_b):
    return [min(a, b) for a, b in zip(list_a, list_b)]


def print_iter(iter):
    #print('iter[0]:', iter[0])
    for pair in iter:
        print(pair)


def splite_signature(row, b, r):
    out = []
    for i in range(b):
        try:
            out.append(((i, tuple(row[1][i*r:(i+1)*r])), row[0]))
        except:
            out.append(((i, tuple(row[1][i*r:])), row[0]))
    return out


def get_pairs(row):
    return [pair for pair in itertools.combinations(list(row[1]), 2)]


def verify_similarity(candidate_pairs, b_us_dict, business_index, threshold=0.05):
    inv_business_index = {v: k for k, v in business_index.items()}
    out_list = []
    bu_dict = b_us_dict
    for pair in candidate_pairs:
        try:
            b0_set = set(bu_dict[pair[0]])
            b1_set = set(bu_dict[pair[1]])
            Jarccard = float(len(b0_set&b1_set))/float(len(b0_set|b1_set))
        except:
            Jarccard = 0
        #print(pair, ':' ,Jarccard)
        if Jarccard >= threshold:
            out_list.append({"b1":inv_business_index[pair[0]], "b2":inv_business_index[pair[1]], "sim":Jarccard})
    return out_list


def output_json(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.close()


def cal_precision_recall(pred_pairs):
    true_list = []
    with open('task1_truepairs.json', 'r') as file:
        for line in file:
            load_dict = json.loads(line)
            true_list.append(tuple(sorted([load_dict['b1'],load_dict['b2']])))
    true_set = set(true_list)
    pred_list = []
    for dict in pred_pairs:
        pred_list.append(tuple(sorted([dict['b1'], dict['b2']])))
    pred_set = set(pred_list)
    precision = len(pred_set&true_set)/len(pred_set)
    recall = len(pred_set&true_set)/len(true_set)
    return precision, recall


def task1(input_file_path, output_path, n, r, b, threshold=0.05):
    '''
    minhash + LSH
    :param input_file_path:
    :param output_path:
    :param n: number of permutation hash function in minhash
    :param r: number of rows in a band
    :param b: number of band
    :return:
    output json file of true pairs and its true Jaccard similarity
    '''
    start_time = time.time()
    sc = SparkContext()
    bu_rdd = sc.textFile(input_file_path).map(lambda row: json.loads(row))

    # tokenize both business_id and user_id
    # business_index
    business_index = bu_rdd.map(lambda row: row['business_id']).distinct().collect()
    business_index = {a: b for a, b in zip(business_index, list(range(len(business_index))))}
    # user_index
    user_index = bu_rdd.map(lambda row: row['user_id']).distinct().collect()
    user_index = {a: b for a, b in zip(user_index, list(range(len(user_index))))}
    user_index_len = len(user_index)

    # creat user-hash rdd
    # output: [user_id, [h1, h2,..., hn]]
    user_hash_index_rdd = bu_rdd.map(lambda row: row['user_id']).distinct().\
        map(lambda row: perm_hash(row, user_index_len, n, user_index))

    # group user by business, this is also the true data
    # output: [b_id, [u_id1, u_id2,...]]
    b_us_dict = bu_rdd.map(lambda row: [business_index[row['business_id']], user_index[row['user_id']]]).groupByKey().\
        map(lambda row: [row[0], list(set(list(row[1])))]).collectAsMap()

    # group business by user
    # output: [u_id, [b_id1, b_id2,...]]
    u_bs_rdd = bu_rdd.map(lambda row: [user_index[row['user_id']], business_index[row['business_id']]]).groupByKey(). \
        map(lambda row: [row[0], list(set(list(row[1])))])

    # construct signature matrix
    # output: [b_id, [min(h1), min(h2), ...]]
    sig_matrix_rdd = u_bs_rdd.leftOuterJoin(user_hash_index_rdd).\
        flatMap(lambda row: [(b_id, row[1][1]) for b_id in row[1][0]]).reduceByKey(min_list)

    # LSH to find candidate pairs
    candidate_pairs = sig_matrix_rdd.flatMap(lambda row: splite_signature(row, b, r)).groupByKey().\
    filter(lambda row: len(row[1])>=2).flatMap(get_pairs).distinct().collect()

    # verify candidate pairs
    true_pairs = verify_similarity(candidate_pairs,  b_us_dict, business_index,threshold=threshold)
    output_json(true_pairs, output_path)
    print("Duriation:%d s" % (time.time() - start_time))
    #precision, recall = cal_precision_recall(true_pairs)
    #print('precision:', precision)
    #print('recall:', recall)


if __name__=='__main__':
    #'''
    start_time = time.time()
    input_file_path = 'resource/asnlib/publicdata/train_review.json'
    output_path = 'task1_out.json'
    #'''
    #input_file_path, output_path = sys.argv[1:]
    # number of permutation hash
    n = 800
    # number of row for each band
    r = 2
    # number of band rb = n
    b = int(math.ceil(float(n)/r))
    task1(input_file_path, output_path, n, r, b)
