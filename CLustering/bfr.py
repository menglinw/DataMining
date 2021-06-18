import sys
import os
import random
import math
import itertools
import time
import csv
import json
'''
file_list: the list of all data file
for index, file in enumerate(file list):
    if index == 0:
        k-means to creat initiation
    elif index == len(file list) -1:
        merge all CS, RS to closest centroid
    else:
        for point in data:
            update CS/DS/RS
    record intermediate result
output cluster result
output intermediate result

'''

class K_means():
    def __init__(self, data: dict, k: int, distance_type: str, max_iter: int, threshold=0.01, seed=2021):
        self.data = data
        self.k = k
        self.distance_type = distance_type
        self.max_iter = int(max_iter)
        self.seed = seed
        self.dimension = len(data[list(data.keys())[0]])
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
            new_cluster_centroid[cluster] = [float(cent)/n for cent in centroid]
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
        cluster_summary = []
        cluster_members = []
        for cluster, members in self.cluster_index.items():
            N = len(members)
            SUM = [0]*self.dimension
            SUMSQ = [0]*self.dimension
            for member in members:
                data_point = self.data[member]
                SUM = [(a + b) for (a, b) in zip(SUM, data_point)]
                SUMSQ = [(a + b ** 2) for (a, b) in zip(SUMSQ, data_point)]
            cluster_summary.append([N, SUM, SUMSQ])
            cluster_members.append(members)
        return cluster_summary, cluster_members


def cal_distance(point1: list, point2: list, dist_type="euclidean", std=None):
    # calculate distance between two point
    if dist_type == "euclidean":
        return float(math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(point1, point2)])))
    elif dist_type == "mahalanobis":
        return float(math.sqrt(sum([((a - b) / sd) ** 2 for (a, b, sd) in zip(point1, point2, std)])))


class RS():
    def __init__(self):
        self.RS_dict = dict()

    def add_points(self, new_points: dict):
        # add new data points to RS
        for index, data in new_points.items():
            self.RS_dict.setdefault(index, [])
            self.RS_dict[index] = data

    def _summarize_cluster(self, cluster_members: list):
        # summarize a cluster, output [N, SUM, SUMSQ]
        N = len(cluster_members)
        dim = len(self.RS_dict[list(self.RS_dict.keys())[0]])
        SUM = [0]*dim
        SUMSQ = [0]*dim
        for i in cluster_members:
            data_point = self.RS_dict[i]
            SUM = [(a+b) for (a, b) in zip(SUM, data_point)]
            SUMSQ = [(a + b**2) for (a, b) in zip(SUMSQ, data_point)]
        return [N, SUM, SUMSQ]

    def withinRS_cluster(self, k):
        # run within RS cluster, return clusters summary and members that have more than 1 point and delete these points from RS
        if len(self.RS_dict) > 5:
            model = K_means(data=self.RS_dict, k=k, distance_type="euclidean", max_iter=1)
            cluster_centroid, cluster_member = model.cluster()
            output_summary = []
            output_members = []
            for members in cluster_member:
                if len(members) > 2:
                    output_summary.append(self._summarize_cluster(members))
                    output_members.append(members)
                    for member in members:
                        self.RS_dict.pop(member)
            return output_summary, output_members
        else:
            return [], []

    def report(self):
        return len(self.RS_dict)

    def output_RS_points(self):
        return list(self.RS_dict.keys())


class CS():
    def __init__(self):
        self.CS_summary_list = []
        self.CS_member_list = []

    def add_clusters(self, add_clusters_summary: list, add_cluster_member: list):
        if len(add_cluster_member) != len(add_clusters_summary):
            print('please check the consistency of the input data')
        else:
            self.CS_summary_list = self.CS_summary_list + add_clusters_summary
            self.CS_member_list = self.CS_member_list + add_cluster_member

    def add_point(self, add_index: int, add_point: list):
        # add a point to CS if the point is close to any cluster
        # otherwise return False indicating fail to add to CS
        for i in range(len(self.CS_summary_list)):
            N = self.CS_summary_list[i][0]
            SUM = self.CS_summary_list[i][1]
            SUMSQ = self.CS_summary_list[i][2]
            centroid = [i/N for i in SUM]
            std = [math.sqrt(b/N - (a)**2) for (a, b) in zip(centroid, SUMSQ)]
            if cal_distance(add_point, centroid, std=std, dist_type="mahalanobis") < 2.5*math.sqrt(len(centroid)):
                # updata the point to CS
                N = N + 1
                SUM = [(a+b) for (a, b) in zip(add_point, SUM)]
                SUMSQ = [(a**2 + b) for (a, b) in zip(add_point, SUMSQ)]
                self.CS_summary_list[i] = [N, SUM, SUMSQ]
                self.CS_member_list[i].append(add_index)
                return True
        return False

    def merge_cluster(self):
        # merge clusters if their distance is closer than threshold
        '''
        for index1, summary1 in enumerate(self.CS_summary_list[:-1]):
            for index2, summary2 in enumerate(self.CS_summary_list[index1+1:]):
                centroid1 = [i/summary1[0] for i in summary1[1]]
                centroid2 = [i/summary2[0] for i in summary2[1]]
                std2 = [math.sqrt(b/summary2[0] - (a)**2) for (a, b) in zip(centroid2, summary2[2])]
                if cal_distance(centroid1, centroid2, std=std2) < 2.5*math.sqrt(len(centroid2)):
                    merged_N = summary1[0] + summary2[0]
                    merged_SUM = [(a+b) for (a, b) in zip(summary1[1], summary2[1])]
                    merged_SUMSQ = [(a+b) for (a, b) in zip(summary1[2], summary2[2])]
                    self.CS_summary_list[index1] = [merged_N, merged_SUM, merged_SUMSQ]
                    self.CS_summary_list.pop(index2)
                    self.CS_member_list[index1] = self.CS_member_list[index1] + self.CS_member_list[index2]
                    self.CS_member_list.pop(index2)
        '''
        for index1, index2 in itertools.combinations(range(len(self.CS_summary_list)), 2):
            n = len(self.CS_summary_list)
            if index1 < n and index2 < n:
                summary1 = self.CS_summary_list[index1]
                summary2 = self.CS_summary_list[index2]
                centroid1 = [i/summary1[0] for i in summary1[1]]
                centroid2 = [i/summary2[0] for i in summary2[1]]
                std2 = [math.sqrt(b/summary2[0] - (a)**2) for (a, b) in zip(centroid2, summary2[2])]
                if cal_distance(centroid1, centroid2, std=std2) < 1.5*math.sqrt(len(centroid2)):
                    merged_N = summary1[0] + summary2[0]
                    merged_SUM = [(a+b) for (a, b) in zip(summary1[1], summary2[1])]
                    merged_SUMSQ = [(a+b) for (a, b) in zip(summary1[2], summary2[2])]
                    self.CS_summary_list[index1] = [merged_N, merged_SUM, merged_SUMSQ]
                    self.CS_summary_list.pop(index2)
                    self.CS_member_list[index1] = self.CS_member_list[index1] + self.CS_member_list[index2]
                    self.CS_member_list.pop(index2)

    def report(self):
        n_cluster = len(self.CS_summary_list)
        n_point = 0
        for summary in self.CS_summary_list:
            n_point += summary[0]
        return n_cluster, n_point

    def output_CS_clusters(self):
        return self.CS_summary_list, self.CS_member_list



class DS():
    def __init__(self):
        self.DS_summary_list = []
        self.DS_member_list = []

    def initialize(self, input_summary_list: list, input_member_list: list):
        if len(input_member_list) == len(input_summary_list):
            self.DS_summary_list = input_summary_list
            self.DS_member_list = input_member_list
        else:
            print('please check your input')

    def add_point(self, add_index: int, add_point: list):
        # add a point to CS if the point is close to any cluster
        # otherwise return False indicating fail to add to CS
        closest_dis = 10*math.sqrt(len(self.DS_summary_list[0][1]))
        closest_cluster = None
        for i in range(len(self.DS_summary_list)):
            N = self.DS_summary_list[i][0]
            SUM = self.DS_summary_list[i][1]
            SUMSQ = self.DS_summary_list[i][2]
            centroid = [i / N for i in SUM]
            std = [math.sqrt(b / N - (a) ** 2) for (a, b) in zip(centroid, SUMSQ)]
            cur_dis = cal_distance(add_point, centroid, std=std, dist_type="mahalanobis")
            if cur_dis < 1 * math.sqrt(len(centroid)) and cur_dis < closest_dis:
                closest_cluster = i
        # updata the point to CS
        if closest_cluster != None:
            N = self.DS_summary_list[closest_cluster][0] + 1
            SUM = self.DS_summary_list[closest_cluster][1]
            SUMSQ = self.DS_summary_list[closest_cluster][2]
            SUM = [(a + b) for (a, b) in zip(add_point, SUM)]
            SUMSQ = [(a ** 2 + b) for (a, b) in zip(add_point, SUMSQ)]
            self.DS_summary_list[closest_cluster] = [N, SUM, SUMSQ]
            self.DS_member_list[closest_cluster].append(add_index)
            return True
        else:
            return False

    def report(self):
        n_cluster = len(self.DS_summary_list)
        n_point = 0
        for summary in self.DS_summary_list:
            n_point += summary[0]
        return n_cluster, n_point

    def output_DS_clusters(self):
        return self.DS_summary_list, self.DS_member_list


class BFR():
    def __init__(self, input_path, n_cluster, out_file1, out_file2):
        '''
        BFR algorithm
        :param input_path:  the folder containing the files of data points
        :param n_cluster: the number of clusters
        :param out_file1: the output file of cluster results
        :param out_file2: the output file of intermediate results
        '''
        self.input_path = input_path
        self.n_cluster = int(n_cluster)
        self.out_file1 = out_file1
        self.out_file2 = out_file2

    def cluster(self):
        file_list = os.listdir(self.input_path)
        # iterate over all file under the input_path
        inter_result = [['round_id', 'nof_cluster_discard', 'nof_point_discard', 'nof_cluster_compression',
                         'nof_point_compression', 'nof_point_retained']]
        for i_file, file in enumerate(sorted(file_list)):
            data_path = os.path.join(self.input_path, file)
            # read in data file as a dictionary
            data = dict()
            for line in open(data_path, "r"):
                data_point = []
                for index, element in enumerate(line.split(',')):
                    if index == 0:
                        data_point.append(int(element))
                    else:
                        data_point.append(float(element))
                data[data_point[0]] = data_point[1:]
            # 2 branches: first round need initialization
            if i_file == 0:
                # select a subset (2000 points) of data to run k-means for initialization
                if len(data) > 100*self.n_cluster:
                    subset_index = set(random.sample(data.keys(), 100*self.n_cluster))
                    sub_data = {k:v for k, v in data.items() if k in subset_index}
                    data = {k:v for k, v in data.items() if k not in subset_index}
                else:
                    sub_data = data
                    data = None
                # step 2
                km_model1 = K_means(data=sub_data, k=5*self.n_cluster, distance_type='euclidean', max_iter=30, threshold=1)
                cluster_summary, cluster_members = km_model1.cluster()
                # print('Step 2 finished')
                # step 3
                outlier = []
                inlier = []
                for cluster in cluster_members:
                    if len(cluster) > 3:
                        inlier += cluster
                    else:
                        outlier += cluster

                #print('Step 3 finished')
                # step 4
                out_data = dict()
                for index in outlier:
                    out_data[index] = sub_data.pop(index)
                # out_data: outlier data, sub_data: inlier data
                km_model2 = K_means(data=sub_data, k=self.n_cluster, distance_type='euclidean', max_iter=30)
                cluster_summary2, cluster_members2 = km_model2.cluster()
                DS_obj = DS()
                DS_obj.initialize(cluster_summary2, cluster_members2)
                sub_data = None
                #print('Step 4 finished')
                # step 5
                n_outlier = len(out_data)
                if n_outlier != 0 :
                    k5 = int(math.ceil(float(n_outlier)/2.0))
                    km_model5 = K_means(data=out_data, k=k5, distance_type='euclidean', max_iter=10)
                    cluster_summary5, cluster_members5 = km_model5.cluster()
                else:
                    cluster_summary5, cluster_members5 = [], []
                CS_member_list = []
                CS_summary_list = []
                RS_idx = set()
                RS_obj = RS()
                CS_obj = CS()
                for i in range(len(cluster_summary5)):
                    if len(cluster_members5[i]) > 1:
                        CS_member_list.append(cluster_members5[i])
                        CS_summary_list.append(cluster_summary5[i])
                    else:
                        RS_idx.add(cluster_members5[i][0])
                RS_points = {k:v for k, v in out_data.items() if k in RS_idx}
                RS_obj.add_points(RS_points)
                CS_obj.add_clusters(CS_summary_list, CS_summary_list)
                #print('Step 5 finished')
                # STEP 7
                if data != None:
                    while len(data) != 0:
                        # print('Data point remaining:', len(data))
                        cur_index, cur_point = data.popitem()
                        add_to_DS = DS_obj.add_point(cur_index, cur_point)
                        add_to_CS = False
                        if not add_to_DS:
                            add_to_CS = CS_obj.add_point(cur_index, cur_point)
                        if not add_to_DS and not add_to_CS:
                            RS_obj.add_points({cur_index:cur_point})

                        if RS_obj.report() > 5*self.n_cluster:
                            RS_to_CS_summary, RS_to_CS_members = RS_obj.withinRS_cluster(self.n_cluster)
                            CS_obj.add_clusters(RS_to_CS_summary, RS_to_CS_members)

            else:
                if data != None:

                    while len(data) != 0:
                        # print('Data point remaining:', len(data))
                        cur_index, cur_point = data.popitem()
                        add_to_DS = DS_obj.add_point(cur_index, cur_point)
                        add_to_CS = False
                        if not add_to_DS:
                            add_to_CS = CS_obj.add_point(cur_index, cur_point)
                        if not add_to_DS and not add_to_CS:
                            RS_obj.add_points({cur_index: cur_point})

                        if RS_obj.report() > 5 * self.n_cluster:
                            RS_to_CS_summary, RS_to_CS_members = RS_obj.withinRS_cluster(self.n_cluster)
                            CS_obj.add_clusters(RS_to_CS_summary, RS_to_CS_members)
            CS_obj.merge_cluster()
            DS_n_cluster, DS_n_point = DS_obj.report()
            CS_n_cluster, CS_n_point = CS_obj.report()
            RS_n_point = RS_obj.report()
            inter_result.append([i_file+1, DS_n_cluster, DS_n_point, CS_n_cluster, CS_n_point, RS_n_point])
            # print([i_file+1, DS_n_cluster, DS_n_point, CS_n_cluster, CS_n_point, RS_n_point])
        self.inter_result = inter_result
        # TODO output
        # iter_result need to be output
        DS_summary_list, DS_member_list = DS_obj.output_DS_clusters()
        CS_summary_list, CS_member_list = CS_obj.output_CS_clusters()
        RS_point_list = RS_obj.output_RS_points()
        # merge CS to DS
        for i in range(len(CS_member_list)):
            nearest_cluster = 0
            nearest_dis = 1000000000000
            for j in range(len(DS_summary_list)):
                centroid1 = [d/CS_summary_list[i][0] for d in CS_summary_list[i][1]]
                centroid2 = [d / DS_summary_list[j][0] for d in DS_summary_list[j][1]]
                cur_dis = cal_distance(centroid1, centroid2, dist_type='euclidean')
                if cur_dis < nearest_dis:
                    nearest_dis = cur_dis
                    nearest_cluster = j
            DS_member_list[nearest_cluster] = DS_member_list[nearest_cluster] + CS_member_list[i]
            N, SUM, SUMSQ = DS_summary_list[nearest_cluster]
            N += CS_summary_list[i][0]
            SUM = [(a+b) for (a,b) in zip(SUM, CS_summary_list[i][1])]
            SUMSQ = [(a + b) for (a, b) in zip(SUMSQ, CS_summary_list[i][2])]
            DS_summary_list[nearest_cluster] = [N, SUM, SUMSQ]
            DS_member_list[nearest_cluster] += CS_member_list[i]
        self.cluster_members = DS_member_list
        self.outlier_members = RS_point_list
        return self.cluster_members, self.outlier_members
        # print(self.cluster_members)

    def export_file(self):
        # export intermediate result
        with open(self.out_file2, 'w+', newline="") as file:
            writer = csv.writer(file)
            for content in self.inter_result:
                writer.writerow(content)
        # export cluster result
        output_dict = dict()
        for cluster, members in enumerate(self.cluster_members):
            for member in members:
                output_dict[member] = cluster
        for out_member in self.outlier_members:
            output_dict[out_member] = -1
        output_dict = {str(k):v for (k, v) in sorted(output_dict.items(), key=lambda d:d[0])}
        with open(self.out_file1, 'w+') as file:
            file.writelines(json.dumps(output_dict))
            file.close()

if __name__ == "__main__":
    start = time.time()
    #'''
    input_path = 'data/test1'
    n_cluster = 10
    out_file1 = 'cluster_result'
    out_file2 = 'intermediate_result'
    #'''
    #input_path, n_cluster, out_file1, out_file2 = sys.argv[1:]
    model = BFR(input_path, n_cluster, out_file1, out_file2)
    model.cluster()
    model.export_file()
    print('Duriation:', time.time() - start)
