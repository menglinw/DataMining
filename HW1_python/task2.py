from pyspark import SparkContext
import json
import sys


class Task2:
    def __init__(self, review_file, business_file, output_file, if_spark, n):
        self.if_spark = if_spark
        self.n = n
        self.output_file = output_file
        self.output = {}

        if self.if_spark == 'spark':
            sc = SparkContext()
            input_review = sc.textFile(review_file)
            input_business = sc.textFile(business_file)
            self.input_review = input_review.map(lambda row: json.loads(row)).\
                map(lambda row: (row['business_id'], row['stars'])).filter(lambda row: row[1]!=None)
            self.input_business = input_business.map(lambda row: json.loads(row)).\
                map(lambda row: (row['business_id'], row['categories'])).filter(lambda row: row[1] != None)
        else:
            self.input_business = {}
            self.input_review = []
            with open(business_file, 'r', encoding="utf-8") as f:
                # 读取所有行 每行会是一个字符串
                for jsonstr in f.readlines():
                    # 将josn字符串转化为dict字典
                    jsondict = json.loads(jsonstr)
                    if jsondict['categories'] != None:
                        self.input_business[jsondict['business_id']] = [cat.strip() for cat in jsondict['categories'].split(',')]

            with open(review_file, 'r', encoding="utf-8") as f:
                # 读取所有行 每行会是一个字符串
                for jsonstr in f.readlines():
                    # 将josn字符串转化为dict字典
                    jsondict = json.loads(jsonstr)
                    self.input_review.append([jsondict['business_id'], jsondict['stars']])

    def cal_avg_stars(self):
        def split_cat(row):
            stars = row[1][0]
            cats = row[1][1].split(',')
            output = []
            for cate in cats:
                output.append([cate.strip(), [stars, 1]])
            return output
        if self.if_spark == 'spark':
            self.output['result'] = self.input_review.join(self.input_business).flatMap(split_cat).\
                reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])).map(lambda row: (row[0], round(row[1][0]/row[1][1], 1))).\
                sortBy(lambda row: (-row[1], row[0]), ascending=True).take(self.n)
        else:
            cat_count = {}
            for review in self.input_review:
                try:
                    for cat in self.input_business[review[0]]:
                        if cat in cat_count.keys():
                            cat_count[cat] = [(cat_count[cat][0] + review[1]), (cat_count[cat][1]+1)]
                        else:
                            cat_count[cat] = [review[1], 1]
                except:
                    pass

            for cat in cat_count.keys():
                cat_count[cat] = round(cat_count[cat][0]/cat_count[cat][1], 1)
            self.output['result'] = sorted(cat_count.items(), key=lambda x: (-x[1], x[0]))[:self.n]

    def save_to_json(self):
        self.cal_avg_stars()
        print(self.output)
        with open(self.output_file, 'w+') as json_out:
            json.dump(self.output, json_out)
        json_out.close()


if __name__ == '__main__':
    (review_file, business_file, output_file, if_spark, n) = sys.argv[1:] #('../data/data/review.json', '../data/data/business.json', 'out2.json', 'no_spark', 10)
    task = Task2(str(review_file), str(business_file), str(output_file), str(if_spark), int(n))
    task.save_to_json()
