from pyspark import SparkContext
import json
import sys
import csv


def read_json(review_file, business_file):
    sc = SparkContext()
    input_review = sc.textFile(review_file)
    input_business = sc.textFile(business_file)
    business_list = input_business.map(lambda row: json.loads(row)).filter(lambda row: row['state']=='NV').\
        map(lambda row: row['business_id']).collect()
    input_review = input_review.map(lambda row: json.loads(row)).filter(lambda row: row['business_id'] in business_list).\
        map(lambda row: (row['user_id'], row['business_id']))
    header = sc.parallelize([('user_id', 'business_id')])
    input_review = header.union(input_review)
    return input_review.collect()

'''
def map_to_csv(row):
    return ','.join(str(v) for v in row)
'''


def write_csv(data, csv_file_path):
    with open(csv_file_path, 'w', newline= '') as out_csv:
        writer = csv.writer(out_csv)
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    review_file = '../../HW1/data/data/review.json'
    business_file = '../../HW1/data/data/business.json'
    review_data = read_json(review_file, business_file)
    write_csv(review_data, 'task2_data.csv')
