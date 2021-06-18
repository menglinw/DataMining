import json
from pyspark import SparkConf, SparkContext
import random
import binascii
import csv
import time
import sys
'''
1. create a hash function generator
    input: number of hash functions
    output: a set of hash functions
2. read in first data
3. n = 10m
    n: number of bit in array --- hash output max
    m: number of element in data
4. for each element in data:
    apply hash function set and collect result in a set, O
5. discard first data, read in second data
6. for each element in data:
    apply hash function anc check existance in set O
7. output result
'''


def gen_hash(k: int, n: int, seed: int):
    '''
    generate a list of hash function
    :param k: number of hash function
    :param n: number of buckets of each hash function
    :param seed: random seed
    :return: a list of (k) hash function, with n buckets each
    '''
    random.seed(seed)
    list_hash = []
    list_a = random.sample(list(range(k*50)), k)
    list_b = random.sample(list(range(k*50)), k)

    def gen_single_hash(a, b, s):
        def out_hash(x):
            return (x*a + b)%s
        return out_hash
    for a, b in zip(list_a, list_b):
        list_hash.append(gen_single_hash(a,b,n))
    return list_hash


def hash_city(row, hash_list):
    city = int(binascii.hexlify(row.encode('utf8')), 16)
    out_index = [hash(city) for hash in hash_list]
    return out_index


def is_in_first(row, first_set, hash_list):
    if row == '' or row == None:
        return 0
    else:
        city = int(binascii.hexlify(row.encode('utf8')), 16)
        cur_index = [hash(city) for hash in hash_list]
        if set(cur_index).issubset(first_set):
            return 1
        else:
            return 0


def output_file(result_list, output_file_path):
    with open(output_file_path, "w+", newline="") as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(result_list)


def task1(first_json_path, second_json_path, output_file_path):
    conf = SparkConf().setMaster('local[3]').setAppName('HW6_task1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    first_rdd = sc.textFile(first_json_path).map(lambda row: json.loads(row)).map(lambda row: row['city']).distinct().\
        filter(lambda row: row != '')
    # m: number of element in stream
    m = first_rdd.count()
    # n: number of bit in array
    n = 10*m
    # k: optimal number of hash function
    k = 7
    hash_list = gen_hash(k, n, 2021)
    # hash each row in rdd and collect all unique index
    first_index_set = set(first_rdd.flatMap(lambda row: hash_city(row, hash_list)).distinct().collect())

    second_rdd = sc.textFile(second_json_path).map(lambda row: json.loads(row)).map(lambda row: row['city'])
    out_list = second_rdd.map(lambda row: is_in_first(row, first_index_set, hash_list)).collect()
    output_file(out_list, output_file_path)


if __name__ == '__main__':
    start = time.time()
    '''
    first_json_path = 'resource/asnlib/publicdata/business_first.json'
    second_json_path = 'resource/asnlib/publicdata/business_second.json'
    output_file_path = 'task1_out.csv'
    '''
    first_json_path, second_json_path, output_file_path = sys.argv[1:]
    task1(first_json_path, second_json_path, output_file_path)
    print('Duriation:', time.time() - start)