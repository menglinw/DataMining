from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import json
import random
import time
import binascii
import math
import sys

class K_means():
    def __init__(self, data: dict, k: int, distance_type: str, max_iter: int, threshold=0.01, seed=2021):
        self.data = data
        self.k = k
        self.distance_type = distance_type
        self.max_iter = int(max_iter)
        self.seed = seed
        self.dimension = 1
        self.threshold = threshold

    def _initiate_clusters(self):
        random.seed(self.seed)
        sample_point = random.sample(self.data.keys(), self.k)
        # store index of data point in each cluster: {cluster#:[point index]}
        self.cluster_index = dict()
        # store the centroid of each cluster: {cluster#:[centroid]}
        self.cluster_centroid = dict()
        # itered set: used to record clustered data point index
        self.remain_set = set(self.data.keys())
        for i, point in enumerate(sample_point):
            self.cluster_index.setdefault(i, [point])
            self.cluster_centroid.setdefault(i, self.data[point])
            self.remain_set.remove(point)
        # print(self.cluster_index)
        # print(self.cluster_centroid)

    def _find_the_nearest_centroid(self, point):
        # given a point, fine the nearest centroid and retur the cluster number
        nearest_dis = cal_distance(point, self.cluster_centroid[0], self.distance_type)
        nearest_cluster = 0
        for cluster, centroid in self.cluster_centroid.items():
            cur_dis = cal_distance(point, centroid, self.distance_type)
            if cur_dis < nearest_dis:
                nearest_cluster = cluster
                nearest_dis = cur_dis
        return nearest_cluster

    def _update_cluster_mumber(self, point_index, target_cluster):
        # update the point_index to its target cluster
        for cluster, index_list in self.cluster_index.items():
            if len(index_list) == 1 and point_index in index_list:
                break
            else:
                if point_index in index_list and cluster != target_cluster:
                    self.cluster_index[cluster].remove(point_index)
                if point_index not in index_list and cluster == target_cluster:
                    self.cluster_index[cluster].append(point_index)
        # update centroid
        '''
        for cluster, index_list in self.cluster_index.items():
            N = len(index_list)
            SUM = [0]*N
            for index in index_list:
                data_point = self.data[index]
                SUM = [(a + b) for (a, b) in zip(SUM, data_point)]
            self.cluster_centroid[cluster] = [i/N for i in SUM]
        '''

    def _cal_cluster_centroid(self):
        # calculate a new centroid based on cluster and its member
        new_cluster_centroid = dict()
        for cluster, members in self.cluster_index.items():
            centroid = [0]*self.dimension
            n = len(members)
            for index in members:
                centroid = [(a + b) for (a, b) in zip(centroid, self.data[index])]
            new_cluster_centroid[cluster] = [i/n for i in centroid]
        return new_cluster_centroid

    def _tell_changed(self, new_cluster_centroid: dict):
        for cluster, centroid in new_cluster_centroid.items():
            if max([(a-b) for (a,b) in zip(centroid, self.cluster_centroid[cluster])]) > self.threshold:
                return True
        return False

    def cluster(self):
        # pick k points (as far as possible)
        # iter over all points
        #     merge to the closest cluster
        # update centroid
        self._initiate_clusters()
        iter_num = 0
        changed = True
        # stoping condition: reaching max iteration or centroid does not change any more
        while iter_num < self.max_iter and changed: # and difference between centroid less than a threshold
            iter_num += 1
            # print('iteration:', iter_num)
            old_cluster_centroid = self.cluster_centroid
            while len(self.remain_set) != 0:
                target_point_index = self.remain_set.pop()
                # print('processing:', self.data[target_point_index])
                # find the nearest centroid
                nearest_cluster = self._find_the_nearest_centroid(self.data[target_point_index])
                # check the existence of target point and
                # assign target point to corresponding cluster index
                self._update_cluster_mumber(target_point_index, nearest_cluster)
                self.cluster_centroid = self._cal_cluster_centroid()
            self.remain_set = set(self.data.keys())
            changed = self._tell_changed(old_cluster_centroid)
        return [i[0] for i in list(self.cluster_centroid.values())]


def cal_distance(point1: list, point2: list, dist_type="euclidean", std=None):
    # calculate distance between two point
    if dist_type == "euclidean":
        return float(math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(point1, point2)])))
    elif dist_type == "mahalanobis":
        return float(math.sqrt(sum([((a - b) / sd) ** 2 for (a, b, sd) in zip(point1, point2, std)])))

def generate_hash(k: int, n: int, seed: int):
    '''
    generate a list of hash function
    :param k: number of hash function
    :param n: number of bit in array
    :param seed: random seed
    :return: a list of (k) hash function, with n buckets each
    '''
    random.seed(seed)
    list_hash = []
    list_a = random.sample(list(range(k*50)), k)
    list_b = random.sample(list(range(k*50)), k)

    def gen_hash(a, b):
        def out_hash(x):
            return format(((x*a + b)%3**n)%(2**n), 'b').zfill(n)
        return out_hash
    for a, b in zip(list_a, list_b):
        list_hash.append(gen_hash(a,b))
    return list_hash


def count_trailing_zeros(city, hash_fun_list):
    '''
    apply a list of hash function to the city and output a list of number of trailing zeros
    :param city:
    :param hash_fun_list:
    :return: a list of trailing zeros
    '''
    n_city = int(binascii.hexlify(city.encode('utf8')), 16)
    out_list = []
    for hash_fun in hash_fun_list:
        bit_list = hash_fun(n_city)
        if len(bit_list.rstrip('0')) == 0:
            trailing_zero = 0
        else:
            trailing_zero = len(bit_list) - len(bit_list.rstrip('0'))
        out_list.append(trailing_zero)
    return out_list


def median(lst: list):
    lst.sort()
    a = len(lst)
    if a % 2 == 0:
        b = lst[int(len(lst)/2)]
        c = lst[int((len(lst)/2)-1)]
        d = (b + c) / 2.0
        return d
    else:
        return lst[int(len(lst)/2)]


def FM_algorithm(RDD):
    cur_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cities = RDD.distinct().collect()
    true_city_num = len(cities)
    hash_fun_list = generate_hash(500, 2000, 2021)
    max_trail_zeros = [-1]*len(hash_fun_list)
    for city in cities:
        hash_result_list = count_trailing_zeros(city, hash_fun_list)
        max_trail_zeros = [max(a, b) for (a, b) in zip(max_trail_zeros, hash_result_list)]
    estimate_dict = {i:[2**n] for i, n in enumerate(max_trail_zeros)}
    model = K_means(data=estimate_dict, k=6, distance_type="euclidean", max_iter=20)
    print(model.cluster())
    estimate_city_num = int(median(model.cluster()))
    file = open(output_file_path, 'a')
    file.write(cur_time_str + ',' + str(true_city_num) + ',' + str(estimate_city_num) + '\n')
    file.close()


def task2(port_num: int, output_file_path: str):
    conf = SparkConf().setMaster('local[3]').setAppName('HW6_task1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    # every 5 second, do a batch
    ssc = StreamingContext(sc, 5)
    input_stream = ssc.socketTextStream('localhost', port_num)
    input_data = input_stream.window(30, 10).map(lambda row: json.loads(row)).map(lambda row: row['city']).\
        filter(lambda row: row != "" and row != None)
    # output
    file = open(output_file_path, 'w')
    file.write("Time,Ground Truth,Estimation\n")
    file.close()
    input_data.foreachRDD(FM_algorithm)
    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":
    #'''
    port_num = 9999
    output_file_path = 'task2.result'
    #'''
    #port_num, output_file_path = sys.argv[1:]
    task2(int(port_num), output_file_path)