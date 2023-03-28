import jieba
import math
import time
import re
import os

def Preproccess(data_root, del_root, aft_del_root):
    data_list_dir = os.listdir(data_root)
    del_list_dir = os.listdir(del_root)
    aft_del_list_dir = os.listdir(aft_del_root)
    data_corpus = []
    del_corpus = []
    aft_del_corpus = []
    cha_count = 0

    #First preprocess
    for del_file_name in  del_list_dir:
        del_file_path = del_root + '/' +str(del_file_name)
        print(del_file_path)
        with open(os.path.abspath(del_file_path), "r", encoding = 'utf-8') as f:
            del_context = f.read()
            del_corpus.extend(del_context.split("\n"))

    
    for data_file_name in  data_list_dir:
        data_file_path = data_root + '/' +str(data_file_name)
        print(data_file_path)
        with open(data_file_path, "r", encoding = 'ANSI') as f:
            data_context = f.read()
            data_context = data_context.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
            data_context = data_context.replace("\n", "")
            data_context = data_context.replace(" ", '')
            data_context = re.sub('\s','',data_context)

            for del_word in del_corpus:
                data_context = data_context.replace(del_word, "")
            cha_count += len(data_context)
            data_corpus.append(data_context)

    data_corpus = jiebaCut(data_corpus)

    # #Second preprocess
    # for aft_del_file_name in  aft_del_list_dir:
    #     aft_del_file_path = aft_del_root + '/' +str(aft_del_file_name)
    #     print(aft_del_file_path)
    #     with open(os.path.abspath(aft_del_file_path), "r", encoding = 'utf-8') as f:
    #         aft_del_context = f.read()
    #         aft_del_corpus.extend(aft_del_context.split("\n"))

    
    # for aft_del_word in aft_del_corpus:
    #     while(True):
    #         if aft_del_word in data_corpus:
    #             print("!")
    #             data_corpus.remove(aft_del_word)
    #         else:
    #             break

    return data_corpus, cha_count

def jiebaCut(corpus):
    new_corpus = []
    for text in corpus:
        words = jieba.cut(text)
        new_corpus.extend(words)
    return new_corpus

def nWordsFre(corpus, N):
    dic_fre = {}
    if N <= len(corpus):
        for i in range(len(corpus)-N):
            tempt = (corpus[i])
            for j in range(1,N):
                tempt = tempt, (corpus[i+j])
            dic_fre[tempt] = dic_fre.get(tempt, 0) + 1
    else:
        print("Error N number.")
    return dic_fre

def calNWordsEntropyModel(corpus, N):
    #Input:N(Positive Integer)
    #Output:N-element model entropy
    if N == 1:
        dic_fre = nWordsFre(corpus, 1)
        n_len = 0
        entropy = 0

        for item in dic_fre.items():
            n_len += item[1]
        for item in dic_fre.items():
            entropy += -item[1] /n_len *math.log(item[1]/n_len, 2)
    else:
        dic_fre_n = nWordsFre(corpus, N)
        dic_fre_n_1 = nWordsFre(corpus, N-1)
        n_len = 0
        entropy = 0

        for item in dic_fre_n.items():
            n_len += item[1]
        # If data base is very big, then n_len equal to n_1_len
        for item in dic_fre_n.items():
            p_joint = item[1] /n_len
            p_cond = item[1] / dic_fre_n_1[item[0][0]]
            entropy += -p_joint *math.log(p_cond, 2)

    return entropy

if __name__ == '__main__':
    data_corpus, cha_count = Preproccess("./data_set", "./first_delete", "./second_delete")
    for i in range(1,4):
        entropy = calNWordsEntropyModel(data_corpus, i)
        print("采用",i,"元语言模型计算得到的信息熵为：", entropy)


