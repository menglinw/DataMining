from pyspark import SparkContext, SparkConf
import json
import os
import random
import math
import time
import itertools

#os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

def combine_dict(row):
    c_dict = {}
    for dict in row[1]:
        c_dict.update(dict)
    return [row[0], c_dict]


def check_commonuser(dict1, dict2):
    if dict1==None or dict2==None:
        return False
    else:
        n = len(set(dict1.keys())&set(dict2.keys()))
        return n >= 3


def cal_similarity(dict1, dict2):
    c_rate_user = list(set(dict1.keys()) & set(dict2.keys()))
    u1_rate = [dict1[user] for user in c_rate_user]
    u2_rate = [dict2[user] for user in c_rate_user]
    mean_1 = sum(u1_rate)/len(u1_rate)
    mean_2 = sum(u2_rate)/len(u2_rate)
    u1_rate = [rate - mean_1 for rate in u1_rate]
    u2_rate = [rate - mean_2 for rate in u2_rate]
    numerator = sum([r1*r2 for r1, r2 in zip(u1_rate, u2_rate)])
    denominator = math.sqrt(sum([r ** 2 for r in u1_rate]) * sum([r ** 2 for r in u2_rate]))
    if denominator != 0:
        w = numerator/denominator
    else:
        w = 0
    return w


def perm_hash(row, m, num_perm):
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
    return [row, [p_hash(row) for p_hash in list_hash]]


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
    return [tuple(sorted(list(pair))) for pair in itertools.combinations(list(row[1]), 2)]


def verify_similarity(candidate_pairs, b_us_dict,  threshold=0.05):
    out_list = []
    bu_dict = b_us_dict
    for pair in candidate_pairs:
        try:
            b0_set = set(bu_dict[pair[0]].keys())
            b1_set = set(bu_dict[pair[1]].keys())
            Jarccard = float(len(b0_set&b1_set))/float(len(b0_set|b1_set))
        except:
            Jarccard = 0
        #print(pair, ':' ,Jarccard)
        if Jarccard >= threshold:
            out_list.append(pair)
    return out_list


def output_json(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.close()


def gen_model(pb_id, data_dict, inv_index, output_path):
    with open(output_path, 'w+') as o_file:
        for pair in itertools.combinations(pb_id,2):
            dict1 = data_dict[pair[0]]
            dict2 = data_dict[pair[1]]
            if check_commonuser(dict1, dict2):
                sim = cal_similarity(dict1, dict2)
                if sim > 0:
                    o_file.writelines(json.dumps({'b1':inv_index[pair[0]],
                                                  'b2':inv_index[pair[1]],
                                                  'sim':sim}) + "\n")
        o_file.close()


def user_gen_model(candidate_pairs, data_dict, inv_index, output_path):
    with open(output_path, 'w+') as o_file:
        for pair in candidate_pairs:
            dict1 = data_dict[pair[0]]
            dict2 = data_dict[pair[1]]
            if check_commonuser(dict1, dict2):
                sim = cal_similarity(dict1, dict2)
                if sim > 0:
                    o_file.writelines(json.dumps({'u1':inv_index[pair[0]],
                                                  'u2':inv_index[pair[1]],
                                                  'sim':sim}) + "\n")
        o_file.close()


def CF_model(bur_rdd, model_file, cf_type):

    # tokenize both business_id and user_id
    # business_index
    business_index = bur_rdd.map(lambda row: row[0]).distinct().collect()
    business_index = {a: b for a, b in zip(business_index, list(range(len(business_index))))}
    inv_business_index = {b:a for a, b in business_index.items()}
    # user_index
    user_index = bur_rdd.map(lambda row: row[1]).distinct().collect()
    user_index = {a: b for a, b in zip(user_index, list(range(len(user_index))))}
    inv_user_index = {b:a for a, b in user_index.items()}
    # tokenize business-id and user-id
    # output: [b_id, {u_id:score, u_id:score...}]
    tbur_rdd = bur_rdd.map(lambda row: [business_index[row[0]], {user_index[row[1]]: row[2]}])

    if cf_type == 'item_based':
        print('start doing item-based CF!')
        ftbur_rdd = tbur_rdd.groupByKey().filter(lambda row: len(list(row[1])) >= 3)
        ftbur_rdd.cache()
        # save {b:{u:s, u:s}} dictionary, eliminate business with less than 3 users
        tbur_dict = ftbur_rdd.map(lambda row: combine_dict(row)).collectAsMap()
        # create possible business-id rdd
        pb_id = ftbur_rdd.map(lambda row: row[0]).distinct().collect()
        # filter with common user >= 3 and calculate similarity
        # output: {'b1':b_id, 'b2':bid, 'sim':w}
        #candidate_pairs = pb_rdd.cartesian(pb_rdd).filter(lambda pair: pair[1] > pair[0]).collect()
            #filter(lambda row: check_commonuser(tbur_dict.get(row[0]), tbur_dict.get(row[1]))).collect()
            #map(lambda row: [row, cal_similarity(tbur_dict.get(row[0]), tbur_dict.get(row[1]))]).\
            #filter(lambda row: row[1] > 0).map(lambda row: {'b1':inv_business_index[row[0][0]],
                                                            #'b2':inv_business_index[row[0][1]],
                                                            #'sim':row[1]}).collect()
        gen_model(pb_id, tbur_dict, inv_business_index, model_file)

    else:
        # user-based CF training
        # n: number of row
        n = 100
        # r: number of row in each band
        r = 2
        # b: number of band
        b = math.ceil(float(n)/2)

        business_index_len = len(business_index)
        # creat business-hash rdd
        # output: [business_id, [h1, h2,..., hn]]
        business_hash_index_rdd = tbur_rdd.map(lambda row: row[0]).distinct(). \
            map(lambda row: perm_hash(row, business_index_len, n))

        # group business by user, this is also the true data
        # output: [u_id, [b_id1, b_id2,...]]
        tbur_rdd = bur_rdd.map(lambda row: [user_index[row[1]], {business_index[row[0]]: row[2]}])
        ftbur_rdd = tbur_rdd.groupByKey().filter(lambda row: len(list(row[1])) >= 3)
        u_bs_dict = ftbur_rdd.map(lambda row: combine_dict(row)).collectAsMap()

        # group user by business
        # output: [b_id, [u_id1,u_id2,...]]
        b_us_rdd = bur_rdd.map(lambda row: [business_index[row[0]], user_index[row[1]]]).groupByKey(). \
            map(lambda row: [row[0], list(set(list(row[1])))])

        # construct signature matrix
        # output: [u_id, [min(h1), min(h2), ...]]
        sig_matrix_rdd = b_us_rdd.leftOuterJoin(business_hash_index_rdd). \
            flatMap(lambda row: [(u_id, row[1][1]) for u_id in row[1][0]]).reduceByKey(min_list)

        # LSH to find candidate pairs
        candidate_pairs = sig_matrix_rdd.flatMap(lambda row: splite_signature(row, b, r)).groupByKey(). \
            filter(lambda row: len(row[1]) >= 2).flatMap(get_pairs).distinct().collect()
        candidate_pairs = verify_similarity(candidate_pairs, u_bs_dict, 0.01)
        user_gen_model(candidate_pairs, u_bs_dict, inv_user_index, model_file)


def content_based_model(u_meta_rdd, content_model_file, user_index, inv_user_index):
    # TODO contentbased model
    content_model = u_meta_rdd.flatMap(lambda row: [(user_index.get(node.strip(), -1),
                                                     (user_index.get(row['user_id'], -99), row['useful']))
                                                    for node in row['friends'].split(',')]).\
        filter(lambda row: row[0] >= 0).\
        groupByKey().map(lambda row: [row[0], list(row[1])]).filter(lambda row: len(row[1]) > 0).\
        mapValues(lambda row: sorted(row, key=lambda elm: elm[1])[-10:]).\
        map(lambda row: {'user': inv_user_index[row[0]], 'ref': [inv_user_index.get(tup[0], -99) for tup in row[1]]}).collect()
    output_json(content_model, content_model_file)


def get_user_bus_avg(bur_rdd, bus_avg_data: dict, user_avg_data: dict, bus_avg_model_file: str, user_avg_model_file: str):
    user_dict = bur_rdd.map(lambda row: (row[1], row[2])).groupByKey().\
        map(lambda row: (row[0], sum(list(row[1]))/len(list(row[1])))).collectAsMap()
    bus_dict = bur_rdd.map(lambda row: (row[0], row[2])).groupByKey().\
        map(lambda row: (row[0], sum(list(row[1]))/len(list(row[1])))).collectAsMap()
    user_dict.update(user_avg_data)
    bus_dict.update(bus_avg_data)
    with open(bus_avg_model_file, 'w') as out_file:
        json.dump(bus_dict, out_file)

    with open(user_avg_model_file, 'w') as out_file:
        json.dump(user_dict, out_file)


if __name__=="__main__":
    start_time = time.time()
    # input file path
    train_file = 'competition/train_review.json'
    user_avg_file = 'competition/user_avg.json'
    bus_avg_file = 'competition/business_avg.json'
    user_meta_file = 'competition/user.json'
    bus_meta_file = 'competition/business.json'

    # output file path
    userCF_model_file = 'userCF.model'
    itemCF_model_file = 'itemCF.model'
    content_model_file = 'content.model'
    bus_avg_model_file = 'business_avg.model'
    user_avg_model_file = 'user_avg.model'

    # read in input data
    conf = SparkConf() \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    # training data
    bur_rdd = sc.textFile(train_file).map(lambda row: json.loads(row)).\
        map(lambda row: [row['business_id'], row['user_id'], row['stars']])
    bur_rdd.cache()
    # bus meta data
    # b_meta_rdd = sc.textFile(bus_meta_file).map(lambda row: json.loads(row))
    # user meta data
    u_meta_rdd = sc.textFile(user_meta_file).map(lambda row: json.loads(row))
    u_meta_rdd.cache()
    # bus avg data
    with open(bus_avg_file) as file:
        bus_avg_dict = json.load(file)
    # user avg data
    with open(user_avg_file) as file:
        user_avg_dict = json.load(file)
    # build user index and inverse user index
    user_index = bur_rdd.map(lambda row: row[1]).distinct().collect()
    user_index = {a: b for a, b in zip(user_index, list(range(len(user_index))))}
    inv_user_index = {b:a for a, b in user_index.items()}


    # fit model
    # fit User-based collaborative filtering model
    CF_model(bur_rdd, userCF_model_file, 'user_based')
    # fit item-based CF model
    CF_model(bur_rdd, itemCF_model_file, 'item_based')
    # get avg user and bus rating
    get_user_bus_avg(bur_rdd, bus_avg_dict, user_avg_dict, bus_avg_model_file, user_avg_model_file)
    # fit content-based model
    content_based_model(u_meta_rdd, content_model_file, user_index, inv_user_index)
    print("Duriation:%d s" % (time.time() - start_time))
