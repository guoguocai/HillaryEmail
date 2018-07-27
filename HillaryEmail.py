import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
import gensim

# 读邮件
def read_emails(file_path):
    df = pd.read_csv(file_path)
    # 去掉原邮件数据中的 NaN 值
    df = df[['Id', 'ExtractedBodyText']].dropna()
    return df

# 通过正则表达式对邮件内的文本内容进行预处理
def clean_email_text(text):
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        # 只留下字母和空格
        if letter.isalpha() or letter==' ':
            pure_text += letter
    # 再把那些去除特殊字符后落单的单词，直接排除(比如 don't -> don t 去t)。
    # 我们就只剩下有意义的单词了。
    text = ' '.join(word for word in pure_text.split() if len(word)>1)
    return text

# 把内容读入 doc_list 形成列表
def df2list(df):
    docs = df['ExtractedBodyText']
    docs = docs.apply(lambda s: clean_email_text(s))
    # [[email1],[email2],...,[email n]]
    doc_list = docs.values
    return doc_list

# LDA 模型构建
def LDA(doc_list):
    # 1.读停止词
    stopwords = []
    with open('resources/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    # print(stopwords)

    # 2.英文分词，形成 gensim 要求的标准语料
    texts = [[word for word in doc.lower().split() if word not in stopwords] for doc in doc_list]
    # print(len(texts))
    # print(texts[0])

    # 3.建立语料库 corpus（用数字来指代单词）
    dictionary = corpora.Dictionary(texts) # id -> word
    corpus = [dictionary.doc2bow(text) for text in texts] # 变成词袋(213,2) 213号单词出现了2次
    # print(corpus[13])

    # 4.建立 LDA 模型，先人工设定 20 个主题
    lda = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 20)
    for i in range(20):
        print('topic = ' + str(i) + '  ' + lda.print_topic(i, topn=5))  # 主题 i 最常用的 topn 个词
        print('----------------------------------------------')
    return lda

def main():
    file_path = "resources/HillaryEmails.csv"
    df = read_emails(file_path)
    doc_list = df2list(df)
    lda = LDA(doc_list)

    # 接下来通过:lda.get_document_topics(bow)
    # 或者:     lda.get_term_topics(word_id)
    # 两个方法，我们可以把新鲜的文本 / 单词，分类成20个主题中的一个。
    # 但是注意，我们这里的文本和单词，都必须得经过同样步骤的文本预处理 + 词袋化，
    # 也就是说，变成数字表示每个单词的形式。

    for j in range(10):
        topic_id = lda.get_term_topics(word_id=j)  # 传入一个新单词（已用数字表示）
        print(topic_id)                            # 得到该单词所属的主题


if __name__ == '__main__':
    main()