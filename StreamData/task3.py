import tweepy
import sys
import random

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api, output_file_path):
        self.api = api
        self.me = api.me()
        self.seq_num = 0
        self.store_list = []
        self.tag_count_dict = dict()
        self.output_file_path = output_file_path

    def on_status(self, status):
        # get tags
        tag_dict_list = status.entities.get('hashtags')
        # if the tag is not empty
        if len(tag_dict_list) != 0:
            # count seq number
            self.seq_num += 1
            # get clean tag list
            tag_list = []
            for dic in tag_dict_list:
                tag = dic.get('text')
                tag_list.append(tag)

            if len(self.store_list) < 100:
                # when the store list < 100
                # go to store list directly
                self.store_list.append(tag_list)
                # update tag count
                for tag in tag_list:
                    self.tag_count_dict.setdefault(tag, 0)
                    self.tag_count_dict[tag] += 1

            else:
                # in or not
                is_in = random.random() < 100.0/self.seq_num
                if is_in:
                    # if in, choose a tweet in store list to replace
                    pop_index = random.sample(range(100), 1)[0]
                    out_tag_list = self.store_list.pop(pop_index)
                    self.store_list.append(tag_list)
                    # update tag count
                    for out_tag in out_tag_list:
                        self.tag_count_dict[out_tag] -= 1

                    for in_tag in tag_list:
                        self.tag_count_dict.setdefault(in_tag, 0)
                        self.tag_count_dict[in_tag] += 1
        # after update, output current top 3 frequencies
        self._output_current_frequency()

        """
        as a new non-empty tweet come,
        if seq_num <= 100:
            directly store tag list to store list
        if seq_num > 100:
            if in:
                random select a element to replace
             
        """
    def _output_current_frequency(self):
        out_list = sorted(self.tag_count_dict.items(), key=lambda kv: (-kv[1], kv[0]))
        sorted_count_list = sorted(list(set(self.tag_count_dict.values())))
        if len(sorted_count_list) <= 3:
            top3_count = sorted_count_list
        else:
            top3_count = sorted_count_list[-3:]
        file = open(self.output_file_path, "a")
        file.write("The number of tweets with tags from the beginning: %d\n" % self.seq_num)
        for key, value in out_list:
            if value in top3_count:
                file.write(key + " : " + str(value) + "\n")
        file.write("\n")
        file.close()




if __name__=="__main__":
    # command parameter
    _, output_file_path = sys.argv[1:]
    # output_file_path = 'task3.result'

    # key and token
    API_key = 'dnOz4e5wnhjlVPFKkqxYuHnwD'
    API_secret_key = 'rUp1aaCIpQ9jpWi34kT3dAXwzWy8UXeUH3ENF8OaHDLKHMxlsu'
    Access_token = '1385467541795917825-RU0Rb3GuihEsFR5hV7sDOGDSQmr113'
    Access_token_secret = 'kgJF1Ayt7ytzaYZnzsQXJFFKQNriqwWmAPo9mVqX2ZoOr'

    # authorization and test
    auth = tweepy.OAuthHandler(API_key, API_secret_key)
    auth.set_access_token(Access_token, Access_token_secret)
    api = tweepy.API(auth)
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during authentication")

    # build stream
    myStreamListener = MyStreamListener(api, output_file_path)
    myStream = tweepy.Stream(auth=auth, listener=myStreamListener)
    myStream.filter(track=['python'], languages=['en'])