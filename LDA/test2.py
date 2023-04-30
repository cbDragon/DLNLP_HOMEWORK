# coding=utf-8         
import os  
import numpy as np
import jieba
import math
import time
import re
import os
import random
import pandas as pd
from sklearn.cluster import KMeans



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

    #Second preprocess
    with open("second_delete\cn_stopwords.txt", "r", encoding="utf-8") as file:
        stop = file.read()
    stop = stop.split()

    
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
            data_context = jiebaCut(data_context)
            data_context = [word for word in data_context if word not in stop] 
            data_corpus.append(data_context)

    data_set = []
    for data in data_corpus:
        cut = int(len(data) // 13)
        for i in range(13):
            data_split = data[i * cut + 1:i * cut + 500]
            data_split2 = data[i * cut + 501:i * cut + 1000]
            data_set.append(data_split2)

    return data_set

def jiebaCut(text):
    # Choose whether use the words split method
    wordFlag = 0
    if wordFlag == 1:
        words = jieba.lcut(text)
    else:
        words = [word for word in text]
    return words

def LDATrain(data_set, topic_num = 40, alpha = 0.01, beta = 0.1, count_max = 40):
    # Initia the topic set
    topic_set = []
    for i in range(topic_num):
        dic = {}
        topic_set.append(dic)

    # Initia the doc's words topic
    topics_doc_word = []

    for txt in data_set:
        topic_doc_word = []
        for word in txt:
            # Random a topic from 0~topic_num-1 for every word in the txt
            t = random.randint(0, topic_num-1)
            topic_doc_word.append(t)
            topic_set[t][word] = topic_set[t].get(word, 0) + 1
        topics_doc_word.append(topic_doc_word)

    # Increase searching speed part
    topics_word_num = []
    doc_word_num = []
    doc_topic_num = []
    for i in range(topic_num):
        topics_word_num.append(sum(topic_set[i].values()))
    count = 0
    for txt in data_set:
        doc_word_num.append(len(txt))

        t_d_t_num = []
        topic_doc_word = topics_doc_word[count]
        for i in range(topic_num):
            t_d_t_num.append(topic_doc_word.count(i))
        doc_topic_num.append(t_d_t_num)
        count += 1

    # Training the LDA model
    count = 0
    print("LDA training start")
    while count <= count_max:
        i = 0
        for txt in data_set:
            topic_doc_word = topics_doc_word[i]
            doc_word_num = len(txt)
            # print("    txt:", i)
            for j in range(len(txt)):
                ori_toc = topic_doc_word[j]
                word = txt[j]

                # Gibbs sampling update the word topic 
                gibbs_p = []
                for k in range(topic_num):
                    p = (topic_set[k].get(word, 0) + alpha)/(topics_word_num[k] + alpha) 
                    p *= (doc_topic_num[i][k] + beta)/(doc_word_num + beta)
                    gibbs_p.append(p)
                gibbs_p = np.array(gibbs_p)
                upd_toc = np.random.choice(topic_num, p = gibbs_p / gibbs_p.sum())

                # Update the LDA model
                topic_doc_word[j] = upd_toc
                topic_set[ori_toc][word] -= 1
                topic_set[upd_toc][word] = topic_set[upd_toc].get(word, 0) + 1

                # Increasing search speed part
                topics_word_num[ori_toc] -= 1
                topics_word_num[upd_toc] += 1

                doc_topic_num[i][ori_toc] -= 1
                doc_topic_num[i][upd_toc] += 1
            i += 1
        count += 1
        print("count:", count)
    print("LDA training complete")

    # Get the distribution of topic in each doc
    doc_topic_prob = np.array(doc_topic_num, dtype= float)
    for i in range(len(data_set)):
        doc_topic_prob[i] = doc_topic_prob[i] / len(data_set[i])
        # print(len(data_set[i]))
    print(doc_topic_prob)

    for i in range(topic_num):
        print('this is topic ', i)
        a = topic_set[i]
        res = {}
        for key, value in a.items():  # 每个topic降序排序
            if value != 0:
                res[key] = value
        res = list(sorted(res.items(), key=lambda x: x[1], reverse=True))
        print(res[:25])  # 输出前25个最多的主题词
    print('result')

    return topic_set, doc_topic_prob

def LDATest(topic_set, test_set, count_max = 40):
    # Initia the doc's words topic
    topics_doc_word = []
    topic_num = len(topic_set)

    for txt in test_set:
        topic_doc_word = []
        for word in txt:
            # Random a topic from 0~topic_num-1 for every word in the txt
            t = random.randint(0, topic_num-1)
            topic_doc_word.append(t)
            topic_set[t][word] = topic_set[t].get(word, 0) + 1
        topics_doc_word.append(topic_doc_word)

    # Increase searching speed part
    topics_word_num = []
    doc_word_num = []
    doc_topic_num = []
    for i in range(topic_num):
        topics_word_num.append(sum(topic_set[i].values()))
    count = 0
    for txt in test_set:
        doc_word_num.append(len(txt))

        t_d_t_num = []
        topic_doc_word = topics_doc_word[count]
        for i in range(topic_num):
            t_d_t_num.append(topic_doc_word.count(i))
        doc_topic_num.append(t_d_t_num)
        count += 1

    # Training the LDA model
    count = 0
    print("LDA testing start")
    while count <= count_max:
        i = 0
        for txt in test_set:
            topic_doc_word = topics_doc_word[i]
            doc_word_num = len(txt)
            # print("    txt:", i)
            for j in range(len(txt)):
                ori_toc = topic_doc_word[j]
                word = txt[j]

                # Gibbs sampling update the word topic 
                gibbs_p = []
                for k in range(topic_num):
                    p = topic_set[k].get(word, 0)/topics_word_num[k]
                    p *= doc_topic_num[i][k]/doc_word_num
                    gibbs_p.append(p)
                gibbs_p = np.array(gibbs_p)
                upd_toc = np.random.choice(topic_num, p = gibbs_p / gibbs_p.sum())

                # Update the LDA model
                topic_doc_word[j] = upd_toc

                # Increasing search speed part
                doc_topic_num[i][ori_toc] -= 1
                doc_topic_num[i][upd_toc] += 1
            i += 1
        count += 1
        print("count:", count)
    print("LDA testing complete")

    # Get the distribution of topic in each doc
    doc_topic_prob = np.array(doc_topic_num, dtype= float)
    for i in range(len(test_set)):
        doc_topic_prob[i] = doc_topic_prob[i] / len(test_set[i])
        # print(len(data_set[i]))
    print(doc_topic_prob)

    return doc_topic_prob

if __name__ == "__main__":
    # data_set = Preproccess("./data_set", "./first_delete", "./second_delete")
    # print("data_set:", data_set)

    # df = pd.DataFrame(data_set)
    # # df.to_csv('data_set_words.csv',index= False, header= False,encoding='utf-8-sig')
    # # df.to_csv('test_set_words.csv',index= False, header= False,encoding='utf-8-sig')
    # # df.to_csv('data_set_char.csv',index= False, header= False,encoding='utf-8-sig')
    # df.to_csv('test_set_char.csv',index= False, header= False,encoding='utf-8-sig')

    # Read the data set with word split
    file_words = pd.read_csv('data_set_words.csv', header= None,engine='python',encoding='utf-8-sig')
    file_words2 = pd.read_csv('test_set_words.csv', header= None,engine='python',encoding='utf-8-sig')
    # file_words = pd.read_csv('data_set_char.csv', header= None,engine='python',encoding='utf-8-sig')
    # file_words2 = pd.read_csv('test_set_char.csv', header= None,engine='python',encoding='utf-8-sig')
    data_set=file_words.values[0::,0::]
    test_set=file_words2.values[0::,0::]
    # print(data_set[0])

    # Initia the parameters
    topic_num = 40 # decide the topic numbers
    alpha = 0.01 # decide the Gibbs sampling parameters
    beta = 0.1

    LDA_par,  data_topic_prob = LDATrain(data_set, topic_num = topic_num, alpha = alpha, beta = beta, count_max = 30)

    test_topic_prob = LDATest(LDA_par, test_set, count_max = 30)

    result = []
    for i in range(len(test_set)):
        pro = []
        dis = 0
        # use eular distance to calculate the difference between
        # each test txt and novel set
        for j in range(len(data_set)):
            dis += math.sqrt(sum((test_topic_prob[i] - data_topic_prob[j])**2)) 

            # each novel is divided into 13 part
            if (j+1) % 13 == 0:
                pro.append(dis)
                dis = 0
        m = pro.index(min(pro))
        result.append(m)
    result = np.array(result)
    result = result.reshape(16, 13)
    print(result)