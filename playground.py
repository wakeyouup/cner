# # a = [1,2]
# #
# # def aa(a):
# #     c = []
# #     c.extend(a)
# #     c.append(3)
# #     print("aa中的a",  a)
# #
# # def bb(b):
# #     print("bb中放入aa之前的a", a)
# #     aa(b)
# #     print("bb中放入aa之后的a", a)
# #
# # bb(a)
#
#
#
# from scipy.stats import wasserstein_distance
# from sklearn import feature_extraction
#
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# import numpy as np
# example1 = "I am very happy for this thing ."
# example2 = "I don't know who you are ."
# example3 = "I am glad to hear that ."
# vectorizer = feature_extraction.text.CountVectorizer()
# # corpuse = [example1, example2, example3]
# #
# # # 先试试一个句子
# # text_counts = vectorizer.fit_transform(corpuse)
# # print(text_counts)
# # text_counts = text_counts.todense()
# # print(text_counts)
# #
# # for i in range(len(text_counts)):
# #     print("第",i,"个：", text_counts[i].tolist()[0])
# #     print("第",i,"个：", np.squeeze(text_counts[i], 0).reshape(15).shape)
# # print(len(text_counts))
# # for x, y in [[0, 1],[0,2]]:
# #     dist = wasserstein_distance(np.array(text_counts[x].tolist()[0]), np.array(text_counts[y].tolist()[0]))
# #     print('文档{}与文档{}的卫士距离{}'.format(x, y, dist))
# #     dist = euclidean_distances(text_counts[x], text_counts[y])
# #     print('文档{}与文档{}的欧式距离{}'.format(x, y, dist))
#
# import pandas as pd
# # squad-v1_1是非段落的
# data_dir = "/root/life_long_QG/qg_data"
# tasks = ["arc-easy","bool-np","mctestQA",  "narrative_qa",  "openbookQA",  "race",   "squad2", "squad-v1_1",
# "arc-hard",  "boolqa",   "dropQA",         "multiRC",   "newsQA",        "quoref",      "ropes"]
# corpus = []
# import os
# for task in tasks:
#     inp_src = pd.read_csv(os.path.join(data_dir + "/" + task, "train.csv"))["input"].tolist()
#     tgt_src = pd.read_csv(os.path.join(data_dir + "/" + task, "train.csv"))["target"].tolist()
#     inp_src = " ".join(inp_src)
#     tgt_src = " ".join(tgt_src)
#     src = inp_src + tgt_src
#     corpus.append(src)
# text_counts = vectorizer.fit_transform(corpus)
# print(text_counts)
# text_counts = text_counts.todense()
# print(text_counts)
# # squadv1和squadv2
# print("squadv1和squadv2")
# print(f"{tasks[6]}与{tasks[7]}的欧式距离", euclidean_distances(text_counts[6], text_counts[7]))
# print(f"{tasks[6]}与{tasks[7]}的w距离", wasserstein_distance(text_counts[6].tolist()[0], text_counts[7].tolist()[0]))
# print(f"{tasks[6]}与{tasks[7]}的欧式距离", cosine_similarity(text_counts[6], text_counts[7]))
# # ropes和squadv1
# print("ropes和squadv1")
# print(f"{tasks[-1]}与{tasks[7]}的欧式距离", euclidean_distances(text_counts[-1], text_counts[7]))
# print(f"{tasks[-1]}与{tasks[7]}的w距离", wasserstein_distance(text_counts[-1].tolist()[0], text_counts[7].tolist()[0]))
# print(f"{tasks[-1]}与{tasks[7]}的欧式距离", cosine_similarity(text_counts[-1], text_counts[7]))
# # abbstractive的两个, narrative_qa dropQA
# print("abbstractive的两个, narrative_qa dropQA")
# print(f"{tasks[3]}与{tasks[10]}的欧式距离", euclidean_distances(text_counts[3], text_counts[10]))
# print(f"{tasks[3]}与{tasks[10]}的w距离", wasserstein_distance(text_counts[3].tolist()[0], text_counts[10].tolist()[0]))
# print(f"{tasks[3]}与{tasks[10]}的欧式距离", cosine_similarity(text_counts[3], text_counts[10]))
# # multi-choice qa arc-easy arc-hard
# print("multi-choice qa arc-easy arc-hard")
# print(f"{tasks[8]}与{tasks[0]}的欧式距离", euclidean_distances(text_counts[8], text_counts[0]))
# print(f"{tasks[8]}与{tasks[0]}的w距离", wasserstein_distance(text_counts[8].tolist()[0], text_counts[0].tolist()[0]))
# print(f"{tasks[8]}与{tasks[0]}的欧式距离", cosine_similarity(text_counts[8], text_counts[0]))
# # multi-choice qa
# print("multi-choice qa")
# print(f"{tasks[5]}与{tasks[0]}的欧式距离", euclidean_distances(text_counts[5], text_counts[0]))
# print(f"{tasks[5]}与{tasks[0]}的w距离", wasserstein_distance(text_counts[5].tolist()[0], text_counts[0].tolist()[0]))
# print(f"{tasks[5]}与{tasks[0]}的欧式距离", cosine_similarity(text_counts[5], text_counts[0]))
# # yes or no qa
# print("yes or no qa")
# print(f"{tasks[9]}与{tasks[-4]}的欧式距离", euclidean_distances(text_counts[9], text_counts[-4]))
# print(f"{tasks[9]}与{tasks[-4]}的w距离", wasserstein_distance(text_counts[9].tolist()[0], text_counts[-4].tolist()[0]))
# print(f"{tasks[9]}与{tasks[-4]}的欧式距离", cosine_similarity(text_counts[9], text_counts[-4]))
# # extra 和abstr
# print("extra 和abstr")
# print(f"{tasks[6]}与{tasks[3]}的欧式距离", euclidean_distances(text_counts[6], text_counts[3]))
# print(f"{tasks[6]}与{tasks[3]}的w距离", wasserstein_distance(text_counts[6].tolist()[0], text_counts[3].tolist()[0]))
# print(f"{tasks[6]}与{tasks[3]}的欧式距离", cosine_similarity(text_counts[6], text_counts[3]))
# # extra 和multi-choice
# print("extra 和multi-choice")
# print(f"{tasks[6]}与{tasks[8]}的欧式距离", euclidean_distances(text_counts[6], text_counts[8]))
# print(f"{tasks[6]}与{tasks[8]}的w距离", wasserstein_distance(text_counts[6].tolist()[0], text_counts[8].tolist()[0]))
# print(f"{tasks[6]}与{tasks[8]}的欧式距离", cosine_similarity(text_counts[6], text_counts[8]))
# # extra 和yes no
# print("extra 和yes no")
# print(f"{tasks[6]}与{tasks[9]}的欧式距离", euclidean_distances(text_counts[6], text_counts[9]))
# print(f"{tasks[6]}与{tasks[9]}的w距离", wasserstein_distance(text_counts[6].tolist()[0], text_counts[9].tolist()[0]))
# print(f"{tasks[6]}与{tasks[9]}的欧式距离", cosine_similarity(text_counts[6], text_counts[9]))
# # import pandas as pd
# # corpus = pd.read_csv("/root/life_long_QG/qg_data/narrative_qa/dev_new.csv")
# # inp_src = corpus["input"].tolist()
# #
# # # 添加</s>
# # inp = [i.strip() + " </s>" for i in inp_src]
# # from transformers import T5Tokenizer
# # tokenizer = T5Tokenizer.from_pretrained("t5-base")
# # a = tokenizer.encode(inp[0])
# # print(a)
# # print(tokenizer.decode(a))


import pandas as pd
data = pd.read_csv("/root/life_long_QG/qg_data/boolqa/train.csv")
print(len(data))
print(sum([len(l.strip().split(" "))for l in data["input"].tolist()]) / len(data))
print(sum([len(l.strip().split(" "))for l in data["target"].tolist()]) / len(data))
data = pd.read_csv("/root/life_long_QG/qg_data/boolqa/dev_new.csv")
print(len(data))
print(sum([len(l.strip().split(" "))for l in data["input"].tolist()]) / len(data))
print(sum([len(l.strip().split(" "))for l in data["target"].tolist()]) / len(data))
data = pd.read_csv("/root/life_long_QG/qg_data/boolqa/test_new.csv")
print(len(data))
print(sum([len(l.strip().split(" "))for l in data["input"].tolist()]) / len(data))
print(sum([len(l.strip().split(" "))for l in data["target"].tolist()]) / len(data))