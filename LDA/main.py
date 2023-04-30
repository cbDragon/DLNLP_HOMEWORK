import math
import jieba
import os  # 用于处理文件路径
import random
import numpy as np
def read_novel(path):  # 读取语料内容
    content = []
    names = os.listdir(path)
    for name in names:
            con_temp = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='ANSI') as f:
                con = f.read()
                con = content_deal(con)
                con = jieba.lcut(con)  # 结巴分词
                con_list = list(con)
                pos = int(len(con)//13) ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
                for i in range(13):
                    con_temp = con_temp + con_list[i*pos:i*pos+500]
                content.append(con_temp)
            f.close()
    return content, names


def read_novel1(path):  # 读取语料内容
    content = []
    names = os.listdir(path)
    for name in names:
            con_temp = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='ANSI') as f:
                con = f.read()
                con = content_deal(con)
                con = jieba.lcut(con)  # 结巴分词
                con_list = list(con)
                pos = int(len(con)//13) ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
                for i in range(13):
                    con_temp = con_temp + con_list[i*pos+501:i*pos+1000]
                content.append(con_temp)
            f.close()
    return content, names



def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
          '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
          '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
    for a in ad:
        content = content.replace(a, '')
    return content


if __name__ == '__main__':
    [data_txt, files] = read_novel("dataSet")
    Topic_All = []  # 每篇文章中的每个词来自哪个topic
    Topic_count = {}  # 每个topic有多少词
    Topic_fre0 = {}; Topic_fre1 = {}; Topic_fre2 = {}; Topic_fre3 = {}
    Topic_fre4 = {}; Topic_fre5 = {}; Topic_fre6 = {}; Topic_fre7 = {}
    Topic_fre8 = {}; Topic_fre9 = {}; Topic_fre10 = {}; Topic_fre11 = {}
    Topic_fre12 = {}; Topic_fre13 = {}; Topic_fre14 = {}; Topic_fre15 = {} # 每个topic的词频
    Doc_count = []  # 每篇文章中有多少个词
    Doc_fre = []  # 每篇文章有多少各个topic的词
    i = 0
    for data in data_txt:
        topic = []
        docfre = {}
        for word in data:
            a = random.randint(0, len(data_txt)-1)  # 为每个单词赋予一个随机初始topic
            topic.append(a)
            if '\u4e00' <= word <= '\u9fa5':
                Topic_count[a] = Topic_count.get(a, 0) + 1  # 统计每个topic总词数
                docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
                exec('Topic_fre{}[word]=Topic_fre{}.get(word, 0) + 1'.format(i, i))  # 统计每个topic的词频
        Topic_All.append(topic)
        docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
        Doc_fre.append(docfre)
        Doc_count.append(sum(docfre))  # 统计每篇文章的总词数
        # exec('print(len(Topic_fre{}))'.format(i))
        i += 1
    Topic_count = list(dict(sorted(Topic_count.items(), key=lambda x: x[0], reverse=False)).values())
    # print(Topic_All[0])
    Doc_fre = np.array(Doc_fre)  # 转为array方便后续计算
    Topic_count = np.array(Topic_count)  # 转为array方便后续计算
    Doc_count = np.array(Doc_count)  # 转为array方便后续计算
    print(Doc_fre)
    print(Topic_count)
    print(Doc_count)
    # print(Topic_fre0)
    Doc_pro = []  # 每个topic被选中的概率
    Doc_pronew = []  # 记录每次迭代后每个topic被选中的新概率
    for i in range(len(data_txt)):
        doc = np.divide(Doc_fre[i], Doc_count[i])
        Doc_pro.append(doc)
    Doc_pro = np.array(Doc_pro)
    print(Doc_pro)
    stop = 0  # 迭代停止标志
    loopcount = 1  # 迭代次数
    while stop == 0:
        i = 0
        for data in data_txt:
            top = Topic_All[i]
            for w in range(len(data)):
                word = data[w]
                pro = []
                topfre = []
                if '\u4e00' <= word <= '\u9fa5':
                    for j in range(len(data_txt)):
                        exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数
                    pro = Doc_pro[i] * topfre / Topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                    m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                    Doc_fre[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                    Doc_fre[i][m] += 1
                    Topic_count[top[w]] -= 1  # 更新每个topic的总词数
                    Topic_count[m] += 1
                    exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) - 1'.format(top[w], top[w]))  # 更新每个topic该词的频数
                    exec('Topic_fre{}[word] = Topic_fre{}.get(word, 0) + 1'.format(m, m))
                    top[w] = m
            Topic_All[i] = top
            i += 1
        print(Doc_fre, 'new')
        print(Topic_count, 'new')
        if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
            for i in range(len(data_txt)):
                doc = np.divide(Doc_fre[i], Doc_count[i])
                Doc_pronew.append(doc)
            Doc_pronew = np.array(Doc_pronew)
        else:
            for i in range(len(data_txt)):
                doc = np.divide(Doc_fre[i], Doc_count[i])
                Doc_pronew[i] = doc
        print(Doc_pro)
        print(Doc_pronew)
        if (Doc_pronew == Doc_pro).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为模型已经训练完毕
            stop = 1
        else:
            Doc_pro = Doc_pronew.copy()
        loopcount += 1
    print(Doc_pronew)  # 输出最终训练的到的每篇文章选中各个topic的概率
    print(loopcount)  # 输出迭代次数
    # print(Doc_pro0)
    # print(Doc_pronew0)
    print('模型训练完毕！')

    [test_txt, files] = read_novel1("dataSet")
    Doc_count_test = []  # 每篇文章中有多少个词
    Doc_fre_test = []  # 每篇文章有多少各个topic的词
    Topic_All_test = []  # 每篇文章中的每个词来自哪个topic
    i = 0
    for data in test_txt:
        topic = []
        docfre = {}
        for word in data:
            a = random.randint(0, len(data_txt) - 1)  # 为每个单词赋予一个随机初始topic
            topic.append(a)
            if '\u4e00' <= word <= '\u9fa5':
                docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
        Topic_All_test.append(topic)
        docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
        Doc_fre_test.append(docfre)
        Doc_count_test.append(sum(docfre))  # 统计每篇文章的总词数
        i += 1
    # print(Topic_All[0])
    Doc_fre_test = np.array(Doc_fre_test)
    Doc_count_test = np.array(Doc_count_test)
    print(Doc_fre_test)
    print(Doc_count_test)
    Doc_pro_test = []  # 每个topic被选中的概率
    Doc_pronew_test = []  # 记录每次迭代后每个topic被选中的新概率
    for i in range(len(test_txt)):
        doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
        Doc_pro_test.append(doc)
    Doc_pro_test = np.array(Doc_pro_test)
    Doc_pro_test = list(Doc_pro_test)
    print(Doc_pro_test)
    stop = 0  # 迭代停止标志
    loopcount = 1  # 迭代次数
    while stop == 0:
        i = 0
        for data in test_txt:
            top = Topic_All_test[i]
            for w in range(len(data)):
                word = data[w]
                pro = []
                topfre = []
                if '\u4e00' <= word <= '\u9fa5':
                    for j in range(len(data_txt)):
                        exec('topfre.append(Topic_fre{}.get(word, 0))'.format(j))  # 读取该词语在每个topic中出现的频数
                    pro = Doc_pro_test[i] * topfre / Topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                    m = np.argmax(pro)  # 认为该词是由上述概率之积最大的那个topic产生的
                    Doc_fre_test[i][top[w]] -= 1  # 更新每个文档有多少各个topic的词
                    Doc_fre_test[i][m] += 1
                    top[w] = m
            Topic_All_test[i] = top
            i += 1
        print(Doc_fre_test, 'new')
        if loopcount == 1:  # 计算新的每篇文章选中各个topic的概率
            for i in range(len(test_txt)):
                doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
                Doc_pronew_test.append(doc)
            Doc_pronew_test = np.array(Doc_pronew_test)
        else:
            for i in range(len(test_txt)):
                doc = np.divide(Doc_fre_test[i], Doc_count_test[i])
                Doc_pronew_test[i] = doc
        print(Doc_pro_test)
        print(Doc_pronew_test)
        if (Doc_pronew_test == Doc_pro_test).all():  # 如果每篇文章选中各个topic的概率不再变化，则认为训练集已分类完毕
            stop = 1
        else:
            Doc_pro_test = Doc_pronew_test.copy()
        loopcount += 1
    print(Doc_pronew)
    print(Doc_pronew_test)
    print(loopcount)
    print('测试集测试完毕！')
    result = []
    for k in range(len(test_txt)):
        pro = []
        for i in range(len(data_txt)):
            dis = 0
            for j in range(len(data_txt)):
                dis += (Doc_pro[i][j] - Doc_pro_test[k][j])**2  # 计算欧式距离
            pro.append(dis)
        m = pro.index(min(pro))
        print(pro)
        result.append(m)
    print(files)
    print(result)

