from pyspark import SparkContext
import json
import sys
import random
import math
import time
import itertools


def output_json(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.close()


def item_cal_stars(row, N, model_dict, inv_business_index, inv_user_index):
    u_id = row[0]
    b_id = row[1][0]
    co_br_list = row[1][1]
    co_br_dict = {br_pair[0]:br_pair[1] for br_pair in co_br_list}
    if b_id in co_br_dict:
        star = co_br_dict[b_id]
    else:
        temp_list = []
        min_sim = 0
        for cb_id in co_br_dict.keys():
            pair = tuple(sorted([b_id, cb_id]))
            pair_in = []
            if pair in model_dict:
                if len(temp_list) < N:
                    temp_list.append([pair, [model_dict[pair], co_br_dict[cb_id]]])
                    min_sim = min([i[1][0] for i in temp_list])
                else:
                    if model_dict[pair] > min_sim:
                        i = 0
                        while i < N:
                            if temp_list[i][1][0] == min_sim:
                                temp_list.pop(i)
                                i = N
                            else:
                                i += 1
                        temp_list.append([pair, [model_dict[pair], co_br_dict[cb_id]]])
                        min_sim = min([i[1][0] for i in temp_list])
        if len(temp_list) == 0:
            star = -1
        else:
            star = sum([i[1][1]*i[1][0] for i in temp_list])/sum([abs(i[1][0]) for i in temp_list])
    if star > 5.0:
        star = 5.0
    return {'user_id': inv_user_index[u_id],
            'business_id': inv_business_index[b_id],
            'stars': star}



def predict(train_file, test_file, model_file, output_file, cf_type, N=3):
    # read training data and tokenize
    sc = SparkContext()
    bur_rdd = sc.textFile(train_file).map(lambda row: json.loads(row)). \
        map(lambda row: [row['business_id'], row['user_id'], row['stars']])

    # tokenize both business_id and user_id
    # business_index
    business_index = bur_rdd.map(lambda row: row[0]).distinct().collect()
    business_index = {a: b for a, b in zip(business_index, list(range(len(business_index))))}
    inv_business_index = {b: a for a, b in business_index.items()}
    # user_index
    user_index = bur_rdd.map(lambda row: row[1]).distinct().collect()
    user_index = {a: b for a, b in zip(user_index, list(range(len(user_index))))}
    inv_user_index = {b: a for a, b in user_index.items()}
    # tokenize business-id and user-id
    # output: [b_id, {u_id:score, u_id:score...}]
    tbur_rdd = bur_rdd.map(lambda row: [business_index[row[0]], user_index[row[1]], row[2]]).cache()

    # read test file and tokenize
    test_rdd = sc.textFile(test_file).map(lambda row: json.loads(row)).\
        filter(lambda row: row['business_id'] in business_index and row['user_id'] in user_index). \
        map(lambda row: [business_index[row['business_id']], user_index[row['user_id']]])
    if cf_type == 'item_based':
        # read model file and tokenize, store as dictionary
        model_dict = sc.textFile(model_file).map(lambda row: json.loads(row)).\
            map(lambda row: [tuple(sorted([business_index[row['b1']], business_index[row['b2']]])), row['sim']]).collectAsMap()

        # make sure test rdd is in the following format
        # [tok(u_id), tok(b_id)]
        item_test_rdd = test_rdd.map(lambda row: [row[1], row[0]])

        # reformat tbur_rdd into the following format
        # [tok(u_id), [[tok(b_id), rate], [tok(b_id), rate]....]]
        item_train_rdd = tbur_rdd.map(lambda row: [row[1], [row[0], row[2]]]).groupByKey().\
            map(lambda row: [row[0], [br_pair for br_pair in row[1]]])

        # join test_rdd and train_rdd, generate output list
        output = item_test_rdd.leftOuterJoin(item_train_rdd).\
            map(lambda row: item_cal_stars(row, N, model_dict, inv_business_index, inv_user_index)).\
            filter(lambda row: row['stars'] != -1).collect()
        output_json(output, output_file)
    else:
        # TODO user-based CF predict
        # read model file and tokenize, store as dictionary
        model_dict = sc.textFile(model_file).map(lambda row: json.loads(row)). \
            map(lambda row: [tuple(sorted([user_index[row['u1']], user_index[row['u2']]])),
                             row['sim']]).collectAsMap()


        # reformat tbur_rdd into the following format
        # [tok(b_id), [[tok(u_id), rate], [tok(u_id), rate]....]]
        train_rdd = tbur_rdd.map(lambda row: [row[0], [row[1], row[2]]]).groupByKey(). \
            map(lambda row: [row[0], [ur_pair for ur_pair in row[1]]])

        # join test_rdd and train_rdd, generate output list
        output = test_rdd.leftOuterJoin(train_rdd). \
            map(lambda row: item_cal_stars(row, N, model_dict, inv_user_index, inv_business_index)). \
            filter(lambda row: row['stars'] != -1).collect()
        output_json(output, output_file)


if __name__=="__main__":
    """
    train_file = 'resource/asnlib/publicdata/train_review.json'
    test_file = 'resource/asnlib/publicdata/test_review.json'
    model_file = 'task3item.model'
    output_file = 'task3user.predict'
    cf_type = 'user_based'
    """
    train_file, test_file, model_file, output_file, cf_type = sys.argv[1:]
    start_time = time.time()
    predict(train_file, test_file, model_file, output_file, cf_type)
    print("Duriation: %d s" % (time.time() - start_time))