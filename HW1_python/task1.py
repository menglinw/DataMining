from pyspark import SparkContext
import json
from datetime import datetime
from operator import add
import sys


class Task1:
    def __init__(self, input_file, output_file, stopwords, y, m, n):
        self.input_file_path = input_file
        self.output_file_path = output_file
        self.stopwords_path = stopwords
        self.y = int(y)
        self.m = m
        self.n = n
        sc = SparkContext()
        input_text = sc.textFile(self.input_file_path)
        self.input_rdd = input_text.map(lambda row: json.loads(row))
        self.output = {}
        stopwords_list = ['(', "[", ",", ".", "!", "?", ":", ";", "]", ")"] + [word.strip() for word in
                                                                               open(self.stopwords_path)]
        self.stopwords_set = set(stopwords_list)

    # A: count the total number of reviews
    def question_a(self):
        self.output['A'] = self.input_rdd.count()

    # B: count the number of reviews in a given year, y
    def question_b(self):
        y = self.y
        self.output['B'] = self.input_rdd.filter(lambda row: y == datetime.strptime(row['date'],
                                                                                    '%Y-%m-%d %H:%M:%S').year).count()

    # C: count the number of distinct users who have writen the reviews
    def question_c(self):
        self.output['C'] = self.input_rdd.map(lambda row: row["user_id"]).distinct().count()

    # D: Top m users who have the largest number of reviews and its count (1pts)
    def question_d(self):
        self.output['D'] = self.input_rdd.map(lambda row: (row["user_id"], 1)).\
            reduceByKey(add).sortBy(lambda row: row[1], ascending=False).take(self.m)

    # E: Top n frequent words in the review text. The words should be in lower cases.
    def question_e(self):
        stopwords_set = self.stopwords_set

        def text_to_pair(row):
            output = []
            for word in row['text'].lower().split():
                if word not in stopwords_set:
                    output.append([word, 1])
            return output
        self.output['E'] = self.input_rdd.flatMap(text_to_pair).\
            reduceByKey(add).sortBy(lambda row: row[1], ascending=False).map(lambda row: row[0]).take(self.n)

    def save_to_json(self):
        self.question_a()
        self.question_b()
        self.question_c()
        self.question_d()
        self.question_e()
        print(self.output)
        with open(self.output_file_path, 'w+') as json_out:
            json.dump(self.output, json_out)
        json_out.close()


if __name__ == '__main__':
    (input_file, output_file, stopwords, y, m, n) = sys.argv[1:] #('../data/data/review.json', 'out.json' , '../data/data/stopwords', 2017, 10, 10)
    task = Task1(str(input_file), str(output_file), str(stopwords), int(y), int(m), int(n))
    task.save_to_json()





