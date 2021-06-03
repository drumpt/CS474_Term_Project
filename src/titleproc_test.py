import utils
import nltk

class TitleProc:
    def __init__(self, data_dir, num_trends):
        self.num_trends = num_trends
        self.dfs, self.tokenized_doc = self.pre_process(data_dir)
        print("ROUND 1 - Preprocess ended")

    def pre_process(self, data_dir):
        jsons = utils.get_json_list_from_data_dir(data_dir)
        dfs_2015, dfs_2016, dfs_2017 = utils.get_dataframe_from_json_list_by_year(jsons)
        td_2015 = dfs_2015["title"].apply(lambda x: nltk.tokenize.word_tokenize(x))
        td_2015 = [[x.lower() for x in y] for y in td_2015]
        td_2016 = dfs_2016["title"].apply(lambda x: nltk.tokenize.word_tokenize(x))
        td_2016 = [[x.lower() for x in y] for y in td_2016]
        td_2017 = dfs_2017["title"].apply(lambda x: nltk.tokenize.word_tokenize(x))
        td_2017 = [[x.lower() for x in y] for y in td_2017]
        # print("pre_process ended")
        return [dfs_2015, dfs_2016, dfs_2017], [td_2015, td_2016, td_2017]


if __name__ == "__main__":
    tproc = TitleProc("../dataset", 10)
