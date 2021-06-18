from pyspark import SparkContext
import json
import sys
from operator import add


class Task3:
    def __init__(self, review_file, output_file, partition_type, n_partitions, n):
        self.output_file = output_file
        self.partition_type = partition_type
        self.n_partitions = n_partitions
        self.n = n
        self.output = {}
        sc = SparkContext()
        self.input_review = sc.textFile(review_file).map(lambda row: json.loads(row)).map(lambda row: (row['business_id'], 1))
        if self.partition_type == 'customized':
            def part_f(key):
                return ord(key[0]) + ord(key[-1])
            self.input_review = self.input_review.partitionBy(self.n_partitions, part_f)

    def get_business(self):
        n = self.n

        def count_partition(iterator):
            yield sum(1 for _ in iterator)
        self.output['n_partitions'] = self.input_review.getNumPartitions()
        self.output['n_items'] = self.input_review.mapPartitions(count_partition).collect()
        self.output['result'] = self.input_review.reduceByKey(add).filter(lambda row: row[1] > n).collect()

    def save_to_json(self):
        self.get_business()
        print(self.output)
        with open(self.output_file, 'w+') as json_out:
            json.dump(self.output, json_out)
        json_out.close()


if __name__ == '__main__':
    (review_file, output_file, partition_type, n_partitions, n) = sys.argv[1:]#('../data/data/review.json', 'out2.json', 'default', 30, 50)
    task = Task3(str(review_file), str(output_file), str(partition_type), int(n_partitions), int(n))
    task.save_to_json()