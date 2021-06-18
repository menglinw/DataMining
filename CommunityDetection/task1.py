from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import os
import sys
from graphframes import *
import itertools
import time

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

def output_text(true_pair_list, output_path):
    with open(output_path, 'w+') as o_file:
        for item in true_pair_list:
            o_file.writelines(str(item)[1:-1] + "\n")
        o_file.close()

def task1(filter_threshold, input_file_path, output_file_path):
    conf = SparkConf().setMaster('local[3]').setAppName('HW4_task1')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    input_text = sc.textFile(input_file_path)
    row_name = input_text.first()
    '''
    user_index = input_text.filter(lambda row: row != row_name).map(lambda row: row.split(',')[0]).distinct().\
        zipWithIndex().collectAsMap()
    inv_user_index = {v:k for k, v in user_index.items()}
    business_index = input_text.filter(lambda row: row != row_name).map(lambda row: row.split(',')[1]).distinct().\
        zipWithIndex().collectAsMap()
    inv_business_index = {v:k for k, v in business_index.items()}
    '''
    input_data_dict = input_text.filter(lambda row: row != row_name).\
        map(lambda row: [row.split(',')[0], row.split(',')[1]]).groupByKey().\
        map(lambda row: [row[0], list(row[1])]).collectAsMap()
    vertex_list = []
    edge_list = []
    for pair in itertools.combinations(input_data_dict.keys(), 2):
        if len(set(input_data_dict[pair[0]])&set(input_data_dict[pair[1]])) >= int(filter_threshold):
            vertex_list.append(pair[0])
            vertex_list.append(pair[1])
            edge_list.append([pair[0], pair[1]])
            edge_list.append([pair[1], pair[0]])
    vertex_list = [[id] for id in list(set(vertex_list))]
    spark = SparkSession.builder.appName('HW4_task1').getOrCreate()
    vertex_df = spark.createDataFrame(vertex_list, ['id'])
    edge_df = spark.createDataFrame(edge_list, ["src", "dst"])
    # Vertex DataFrame: A vertex DataFrame should contain a special column named "id" which specifies unique IDs for each vertex in the graph.
    # Edge DataFrame: An edge DataFrame should contain two special columns: "src" (source vertex ID of edge) and "dst" (destination vertex ID of edge).
    g = GraphFrame(vertex_df, edge_df)
    result = g.labelPropagation(maxIter=5)
    #print(result.rdd.take(10))

    output_list = result.rdd.map(lambda row: (row[1], row[0])).groupByKey().map(lambda row: sorted(list(row[1]))) \
        .sortBy(lambda row: (len(row), row[0])).collect()

    output_text(output_list, output_file_path)
if __name__ == "__main__":
    '''
    filter_threshold = 7
    input_file_path = "resource/asnlib/publicdata/ub_sample_data.csv"
    output_file_path = "task1.out"
    '''
    filter_threshold, input_file_path, output_file_path = sys.argv[1:4]
    start = time.time()
    task1(filter_threshold, input_file_path, output_file_path)
    print("Duriation: %d s" % (time.time() - start))








