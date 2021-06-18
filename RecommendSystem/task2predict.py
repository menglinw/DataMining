import json
import math
import time
import sys

def read_test(path):
    out = []
    with open(path, 'r') as file:
        for line in file:
            temp_dict = json.loads(line)
            out.append([temp_dict['user_id'], temp_dict['business_id']])
    return out


def output_json(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.close()


def read_model(path):
    business_out = {}
    user_out = {}
    with open(path, 'r') as file:
        out = business_out
        for line in file:
            temp_dict = json.loads(line)
            if 'user' in temp_dict:
                out = user_out
            #print(temp_dict)
            out.update(temp_dict)
    return business_out, user_out


def cal_cosine_sim(vector1, vector2):
    c_sim = len(set(vector1)&set(vector2))/math.sqrt(len(vector1)*len(vector2))
    return c_sim

def verify_bu_pairs(test_data_path, model_path, output_path):
    test_dat = read_test(test_data_path)
    business_dict, user_dict = read_model(model_path)
    out = []
    for user_id, busi_id in test_dat:
        try:
            c_sim = cal_cosine_sim(business_dict[busi_id], user_dict[user_id])
        except:
            c_sim = 0
        if c_sim >= 0.01:
            out.append({"user_id":user_id, "business_id":busi_id, "sim":c_sim})
    output_json(out, output_path)


if __name__ == '__main__':
    start = time.time()
    #'''
    test_data_path = 'resource/asnlib/publicdata/test_review.json'
    model_path = 'task2.model'
    output_path = "task2.predict"
    #'''
    #test_data_path, model_path, output_path = sys.argv[1:]
    verify_bu_pairs(test_data_path, model_path, output_path)
    print("Duriation: %d s" % (time.time() - start))
