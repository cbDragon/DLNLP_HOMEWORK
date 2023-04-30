# # coding=utf-8         
# import os  
# import sys
# import numpy as np
# import matplotlib
# import scipy
# import matplotlib.pyplot as plt
# from sklearn import feature_extraction  
# from sklearn.feature_extraction.text import TfidfTransformer  
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer 
# import jieba
# import re
 

# def Preproccess(data_root, del_root, aft_del_root):
#     data_list_dir = os.listdir(data_root)
#     del_list_dir = os.listdir(del_root)
#     aft_del_list_dir = os.listdir(aft_del_root)
#     data_corpus = []
#     del_corpus = []
#     aft_del_corpus = []
#     cha_count = 0

#     #First preprocess
#     for del_file_name in  del_list_dir:
#         del_file_path = del_root + '/' +str(del_file_name)
#         print(del_file_path)
#         with open(os.path.abspath(del_file_path), "r", encoding = 'utf-8') as f:
#             del_context = f.read()
#             del_corpus.extend(del_context.split("\n"))

    
#     for data_file_name in  data_list_dir:
#         data_file_path = data_root + '/' +str(data_file_name)
#         print(data_file_path)
#         with open(data_file_path, "r", encoding = 'ANSI') as f:
#             data_context = f.read()
#             data_context = data_context.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
#             data_context = data_context.replace("\n", "")
#             data_context = data_context.replace(" ", '')
#             data_context = re.sub('\s','',data_context)

#             for del_word in del_corpus:
#                 data_context = data_context.replace(del_word, "")
#             cha_count += len(data_context)
#             data_context = jiebaCut(data_context)
#             data_corpus.append(data_context)

#     data_set = []
#     for data in data_corpus:
#         cut = int(len(data) // 13)
#         for i in range(13):
#             data_split = data[i * cut + 1:i * cut + 500]
#             data_set.append(data_split)

#     return data_set

# def jiebaCut(text):
#     # Choose whether use the words split method
#     wordFlag = 1
#     if wordFlag == 1:
#         words = jieba.lcut(text)
#     else:
#         words = [word for word in text]
#     return words


# if __name__ == "__main__":
 
 
#     # #存储读取语料 一行预料为一个文档 
#     # corpus = []
#     # for line in open('test.txt', 'r').readlines():
#     #     #print line
#     #     corpus.append(line.strip())
#     # #print corpus

#     data_set = Preproccess("./data_set", "./first_delete", "./second_delete")
#     print("data_set:", data_set)

#     #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
#     vectorizer = CountVectorizer()
#     print (vectorizer)
 
#     X = vectorizer.fit_transform(data_set)
#     analyze = vectorizer.build_analyzer()
#     weight = X.toarray()
 
#     print (len(weight))
#     print (weight[:5, :5])
 
#     #LDA算法
#     print ('LDA:')
#     import numpy as np
#     import lda
#     import lda.datasets
#     model = lda.LDA(n_topics=2, n_iter=500, random_state=1)
#     model.fit(np.asarray(weight))     # model.fit_transform(X) is also available
#     topic_word = model.topic_word_    # model.components_ also works
 
#     #文档-主题（Document-Topic）分布
#     doc_topic = model.doc_topic_
#     print("type(doc_topic): {}".format(type(doc_topic)))
#     print("shape: {}".format(doc_topic.shape))
 
#     #输出前10篇文章最可能的Topic
#     label = []      
#     for n in range(10):
#         topic_most_pr = doc_topic[n].argmax()
#         label.append(topic_most_pr)
#         print("doc: {} topic: {}".format(n, topic_most_pr))
        
#     #计算文档主题分布图
#     import matplotlib.pyplot as plt  
#     f, ax= plt.subplots(6, 1, figsize=(8, 8), sharex=True)  
#     for i, k in enumerate([0, 1, 2, 3, 8, 9]):  
#         ax[i].stem(doc_topic[k,:], linefmt='r-',  
#                    markerfmt='ro', basefmt='w-')  
#         ax[i].set_xlim(-1, 2)     #x坐标下标
#         ax[i].set_ylim(0, 1.2)    #y坐标下标
#         ax[i].set_ylabel("Prob")  
#         ax[i].set_title("Document {}".format(k))  
#     ax[5].set_xlabel("Topic")
#     plt.tight_layout()
#     plt.show()  



import random

for i in range(100):
    print(random.randint(0, 5))