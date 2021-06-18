from pyspark import SparkContext
import json
import sys
from collections import Counter
import math
from operator import add
import time


def text_procesing(row, stopword_set, b_index_dict):
    '''

    :param row: input row
    :param stopword_set:
    :param b_index_dict:
    :return:
    [tokenized business_id, {word:count}]
    '''
    word_count = {}
    word_control_set = set(list('abcdefghijklmnopqrstuvwxyz '))
    clean_str = ''
    for element in row['text'].lower():
        if element in word_control_set:
            clean_str += element
    review_list = filter(lambda word: word not in stopword_set, clean_str.split())
    for word in review_list:
        word_count.setdefault(word, 0)
        word_count[word] += 1
    return [b_index_dict[row['business_id']], word_count]



def user_text_procesing(row, featureword_list, u_index_dict):
    '''

    :param row: input row
    :param stopword_set:
    :param u_index_dict:
    :return:
    [user_id, unique word list]
    '''
    word_control_set = set(list('abcdefghijklmnopqrstuvwxyz '))
    clean_str = ''
    for element in row['text'].lower():
        if element in word_control_set:
            clean_str += element
    clean_str = list(set(clean_str.split()))
    review_list = [featureword_list.index(word) for word in clean_str if word in featureword_list]

    return [u_index_dict[row['user_id']], list(review_list)]


def combine_reviews(wordcount1, wordcount2):
    return dict(Counter(wordcount1) + Counter(wordcount2))


def user_combine_reviews(wordlist1, wordlist2):
    return list(set(wordlist1 + wordlist2))


def count_document(word, comb_br_dict):
    count = 0
    for b_id, item in comb_br_dict.items():
        if word in item:
            count += 1
    return count


def cal_IDF(unique_word_list, comb_br_dict):
    N = len(comb_br_dict)
    IDF_dict = {}
    for word in unique_word_list:
        n = count_document(word, comb_br_dict)
        IDF_dict[word] = math.log(int(N)/n, 2)
    return IDF_dict

'''
def cal_TFIDF(row, IDF_dict):
    TI_dict = {}
    maxn = max(row[1].values())
    for key, value in row[1].items():
        if key in IDF_dict:
            TI_dict[key] = float(value)/maxn*IDF_dict[key]
    TI_list = sorted(TI_dict.values(), reverse=True)
    if len(TI_list) <= 200:
        return [row[0], list(TI_dict.keys())]
    else:
        threshold = TI_list[199]
        return [row[0], [key for key, value in TI_dict.items() if value >= threshold]]
'''


def cal_TFIDF(row, IDF_dict):
    TI_list = []
    maxn = max(row[1].values())
    for key, value in row[1].items():
        if key in IDF_dict:
            TI_list.append([key, float(value)/maxn*IDF_dict[key]])
    return TI_list


def featurewords_to_vector(row, feature_word_list):
    return [row[0], [feature_word_list.index(word) for word in row[1] if word in feature_word_list]]


def output_json(user_profile_list, business_profile_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in business_profile_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.writelines(json.dumps({'user':"start"}) + "\n")
        for item in user_profile_list:
            o_file.writelines(json.dumps(item) + "\n")
        o_file.close()


def create_profile(review_path, model_path, stopword_path):
    sc = SparkContext()
    input_text = sc.textFile(review_path).map(lambda row: json.loads(row))
    input_text.cache()
    stopword_list = [i.strip() for i in open(stopword_path)]
    stopword_set = set(stopword_list)

    # create business_id index
    b_index_dict = input_text.map(lambda row: row['business_id']).distinct().zipWithIndex().collectAsMap()
    b_inverse_index_dict = {b: a for a, b in b_index_dict.items()}

    # create user_id index
    u_index_dict = input_text.map(lambda row: row['user_id']).distinct().zipWithIndex().collectAsMap()
    u_inverse_index_dict = {b: a for a, b in u_index_dict.items()}
    point1 = time.time()
    # input json text
    # output [tokenized business_id, {word: count}]
    single_br_rdd = input_text.map(lambda row: text_procesing(row, stopword_set, b_index_dict))

    # create word index
    # word_index = single_br_rdd.flatMap(lambda row: [key for key, value in row[1].items()]).distinct().zipWithIndex().collectAsMap()
    # tokenize word
    # single_br_rdd = single_br_rdd.map(lambda row: [row[0], {k:v for k, v in row[1].items()}])
    # concatenating reviews of the same business
    comb_br_rdd = single_br_rdd.reduceByKey(combine_reviews)
    comb_br_rdd.cache()
    comb_br_dict = comb_br_rdd.collectAsMap()
    # calculate total word count and rare word count threshold
    totalword_count = comb_br_rdd.map(lambda row: sum(row[1].values())).reduce(add)
    rare_threshold = totalword_count*0.000001
    # get unique word list, eliminate rare word
    unique_word_list = comb_br_rdd.flatMap(lambda row: [[k, v] for k, v in row[1].items()]).reduceByKey(add).\
        filter(lambda row: row[1] > rare_threshold).map(lambda row: row[0]).distinct().collect()
        #comb_br_rdd.flatMap(lambda row: [key for key,value in row[1].items() if value > rare_threshold]).\
        #distinct().collect()
    # calculate IDF of each unique word
    IDF_dict = cal_IDF(unique_word_list, comb_br_dict)
    # calculate TFIDF of each word in each business_id and take the top 200 word with highest TFIDF score
    # feature_word_list output:[b_id, [word1, word2,...,word200]]
    feature_word_list = comb_br_rdd.flatMap(lambda row: cal_TFIDF(row, IDF_dict)).reduceByKey(max).\
        sortBy(lambda row: row[1], ascending=False).map(lambda row: row[0]).take(200)

    # not really necessary to convert to a actual vector
    # output [[business_id, [word_index1, word_index2]]]
    business_profile_list = comb_br_rdd.map(lambda row: [b_inverse_index_dict[row[0]],list(row[1].keys())]).\
        map(lambda row: featurewords_to_vector(row, feature_word_list)).filter(lambda row: len(row[1])!= 0).\
        map(lambda row: {row[0]: row[1]}).collect()

    # now I have feature_vector of business_id, feature_word_index, b_index_dict (business index)
    # create feature vector of user_id, in the same feature_word_index dimension
    # create combined review list for each user_id
    # output [user_id, [word1, word2,...]]
    user_profile_list = input_text.map(lambda row: user_text_procesing(row, feature_word_list, u_index_dict)).\
        filter(lambda row: len(row[1]) != 0).reduceByKey(user_combine_reviews).\
        map(lambda row: {u_inverse_index_dict[row[0]]: row[1]}).collect()

    output_json(user_profile_list, business_profile_list, model_path)

if __name__=="__main__":
    #'''
    review_path = 'resource/asnlib/publicdata/train_review.json'
    stopword_path = 'resource/asnlib/publicdata/stopwords'
    model_path = 'task2.model'
    #'''
    start = time.time()
    #review_path, model_path, stopword_path = sys.argv[1:]
    create_profile(review_path, model_path, stopword_path)
    print("Duriation: %d s" % (time.time()-start))