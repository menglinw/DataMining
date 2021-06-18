from pyspark import SparkContext, SparkConf
import json
import os
import sys
import time

#os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


def item_cal_stars(row, N, model_dict, inv_business_index, inv_user_index, u_avg_model, b_avg_model, u_avg_rate, b_avg_rate):
    b_id = row[0]
    u_id = row[1][0]
    co_ur_list = row[1][1]
    co_ur_dict = {br_pair[0]:br_pair[1] for br_pair in co_ur_list}
    if u_id in co_ur_dict:
        star = co_ur_dict[u_id]
    else:
        temp_list = []
        min_sim = 0
        for cu_id in co_ur_dict.keys():
            pair = tuple(sorted([u_id, cu_id]))
            pair_in = []
            if pair in model_dict:
                if len(temp_list) < N:
                    temp_list.append([pair, [model_dict[pair], co_ur_dict[cu_id]]])
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
                        temp_list.append([pair, [model_dict[pair], co_ur_dict[cu_id]]])
                        min_sim = min([i[1][0] for i in temp_list])
        if len(temp_list) == 0:
            star = u_avg_rate
        else:
            star = 0.8* sum([i[1][1]*i[1][0] for i in temp_list])/sum([abs(i[1][0]) for i in temp_list])+ \
                   0.1*u_avg_model.get(u_id, u_avg_rate) + 0.1*b_avg_model.get(b_id, b_avg_rate)
    if star > 5.0:
        star = 5.0
    if star < 0.0:
        star = 0.0
    return {'user_id': u_id,
            'business_id': b_id,
            'stars': star}


def new_item_cal_stars(row, inv_user_index, inv_business_index, u_avg_model, b_avg_model, u_avg_rate, b_avg_rate, cont_model):
    u_id = row[1]
    b_id = row[0]
    if isinstance(u_id, str) and isinstance(b_id, str):
        rate = (b_avg_rate + u_avg_rate)/2
    elif isinstance(u_id, str) and isinstance(b_id, int):
        rate = 0.2*u_avg_rate + 0.8*b_avg_model.get(b_id, b_avg_rate)
    else: # user seen, business not seen
        friend_list = cont_model.get(u_id, [])
        if len(friend_list) == 0:
            friend_avg = u_avg_rate
        else:
            sum_score = 0
            for friend in friend_list:
                sum_score += u_avg_model.get(friend, u_avg_rate)
            friend_avg = sum_score/len(friend_list)
        rate = 0.8*friend_avg + 0.2*b_avg_rate
    return {'user_id': inv_user_index.get(u_id, u_id),
            'business_id': inv_business_index.get(b_id, b_id),
            'stars': rate}


def output_json(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.close()


def predict(test_file, output_file):
    # model file
    userCF_model_file = 'userCF.model'
    itemCF_model_file = 'itemCF.model'
    content_model_file = 'content.model'

    # input data file
    bus_avg_model_file = 'competition/business_avg.json'
    user_avg_model_file = 'competition/user_avg.json'
    train_file = 'competition/train_review.json'

    # loading data
    # training data
    conf = SparkConf() \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    bur_rdd = sc.textFile(train_file).map(lambda row: json.loads(row)). \
        map(lambda row: [row['business_id'], row['user_id'], row['stars']])
    bur_rdd.cache()
    # build user index and inverse user index
    user_index = bur_rdd.map(lambda row: row[1]).distinct().collect()
    user_index = {a: b for a, b in zip(user_index, list(range(len(user_index))))}
    inv_user_index = {b: a for a, b in user_index.items()}
    # business_index
    business_index = bur_rdd.map(lambda row: row[0]).distinct().collect()
    business_index = {a: b for a, b in zip(business_index, list(range(len(business_index))))}
    inv_business_index = {b: a for a, b in business_index.items()}

    # loading model as dictionary
    # read model file and tokenize, store as dictionary
    itemCF_model = sc.textFile(itemCF_model_file).map(lambda row: json.loads(row)). \
        map(lambda row: [tuple(sorted([business_index[row['b1']], business_index[row['b2']]])), row['sim']]).collectAsMap()
    # user_CF model, dict
    CF_model = sc.textFile(userCF_model_file).map(lambda row: json.loads(row)).\
        map(lambda row: [tuple(sorted([user_index[row['u1']], user_index[row['u2']]])), row['sim']]).collectAsMap()
    # content model, dict
    cont_model = sc.textFile(content_model_file).map(lambda row: json.loads(row)).\
        map(lambda row: [user_index[row['user']], [user_index[u] for u in row['ref']]]).collectAsMap()
    # user avg model, dict
    with open(user_avg_model_file) as file:
        u_avg_model = json.load(file)
    u_avg_model = {user_index[user]:rate for user, rate in u_avg_model.items() if user in user_index}
    # business avg model, dict
    with open(bus_avg_model_file) as file:
        b_avg_model = json.load(file)
    b_avg_model = {business_index[bus]:rate for bus, rate in b_avg_model.items() if bus in business_index}
    b_avg_rate = sum(b_avg_model.values())/len(b_avg_model)
    u_avg_rate = sum(u_avg_model.values())/len(u_avg_model)

    # read test file
    test_rdd = sc.textFile(test_file).map(lambda row: json.loads(row)).\
        map(lambda row: [business_index.get(row['business_id'], row['business_id']),
                         user_index.get(row['user_id'], row['user_id'])])
    # seen_pair, both user and business are in the train set
    seen_pair = test_rdd.filter(lambda row: isinstance(row[0], int) and isinstance(row[1], int))
    new_pair = test_rdd.filter(lambda row: not (isinstance(row[0], int) and isinstance(row[1], int)))

    # predict with user based CF model
    tbur_rdd = bur_rdd.map(lambda row: [business_index[row[0]], user_index[row[1]], row[2]])
    # reformat tbur_rdd into the following format
    # [tok(b_id), [[tok(u_id), rate], [tok(u_id), rate]....]]
    u_train_rdd = tbur_rdd.map(lambda row: [row[0], [row[1], row[2]]]).groupByKey(). \
        map(lambda row: [row[0], [ur_pair for ur_pair in row[1]]])

    user_seen_output = seen_pair.leftOuterJoin(u_train_rdd). \
        map(lambda row: item_cal_stars(row, 5, CF_model, inv_business_index, inv_user_index,
                                       u_avg_model, b_avg_model, u_avg_rate, b_avg_rate)).\
        map(lambda row: [(row['user_id'], row['business_id']), row['stars']])

    # predict with item based CF model
    item_test_rdd = seen_pair.map(lambda row: [row[1], row[0]])
    item_train_rdd = tbur_rdd.map(lambda row: [row[1], [row[0], row[2]]]).groupByKey(). \
        map(lambda row: [row[0], [br_pair for br_pair in row[1]]])
    item_seen_output = item_test_rdd.leftOuterJoin(item_train_rdd). \
        map(lambda row: item_cal_stars(row, 5, itemCF_model, inv_user_index, inv_business_index,
                                       b_avg_model, u_avg_model, b_avg_rate, u_avg_rate)).\
        map(lambda row: [(row['business_id'], row['user_id']), row['stars']])

    seen_output = user_seen_output.leftOuterJoin(item_seen_output).map(lambda row:
                                                                       {'user_id':inv_user_index[row[0][0]],
                                                                        'business_id':inv_business_index[row[0][1]],
                                                                        'stars': min(5.0, 1.52930944 + \
                                                                              0.2414251*row[1][0] + 0.3575569*row[1][1])
                                                                        })
    # predict with content based model
    new_output = new_pair.map(lambda row: new_item_cal_stars(row, inv_user_index, inv_business_index, u_avg_model,
                                                             b_avg_model, u_avg_rate, b_avg_rate, cont_model))
    output = seen_output.union(new_output).collect()
    output_json(output, output_file)

if __name__ == '__main__':
    start = time.time()
    # input argument
    #test_file, output_file = sys.argv[1:]
    test_file, output_file = 'competition/test_review.json', 'predict.result'
    predict(test_file, output_file)
    print('Duriation:', time.time() - start, 's')

