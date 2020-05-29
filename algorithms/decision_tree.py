import random
import numpy as np


# 获得当前数据集的熵值
def get_entropy(data_set):
    if len(data_set) == 0:
        return 0
    label_tags = data_set[:, -1]
    label_length = len(data_set[:, -1])
    entropy = 0
    if sum(label_tags) / label_length > 0:
        entropy -= (sum(label_tags) / label_length) * np.log2(sum(label_tags) / label_length)
    if (1 - sum(label_tags) / label_length) > 0:
        entropy -= (1 - sum(label_tags) / label_length) * np.log2(1 - sum(label_tags) / label_length)
    return entropy


# 根据下标索引indexArray生成一个大小为m的子数据集
def get_sub_data_set(data_set, index_array, m, n):
    sub_data_set = np.zeros((m, n))
    for i in range(m):
        sub_data_set[i] = data_set[index_array[i]].T
    return sub_data_set


# bi-partition二分法求连续属性的划分T
def caculate_t(data_set, feature_tags, entropy, n):
    feature_values = list(set(feature_tags))
    # 如果只有一个值，gain=0
    feature_values.sort()
    # temp_t = 0
    t = 0
    gain = 0
    sub_set_dict = {}
    for i in range(len(feature_values)):
        if(i == len(feature_values)-1) and (len(feature_values) > 1):
            break

        if len(feature_values) > 1:
            temp_t = (feature_values[i]+feature_values[i+1])/2
        else:
            temp_t = feature_values[i]
        no_more_than_t = np.where(feature_tags <= temp_t)
        more_than_t = np.where(feature_tags > temp_t)
        no_more_than_set = get_sub_data_set(data_set, no_more_than_t[0], len(no_more_than_t[0]), n)
        more_than_set = get_sub_data_set(data_set, more_than_t[0], len(more_than_t[0]), n)
        temp_gain = entropy - (len(no_more_than_t[0])/len(data_set))*get_entropy(no_more_than_set)-\
                   (len(more_than_t[0])/len(data_set))*get_entropy(more_than_set)
        if temp_gain >= gain:
            gain = temp_gain
            t = temp_t
            sub_set_dict['noMoreThanSetKey'] = no_more_than_set
            sub_set_dict['moreThanSetKey'] = more_than_set
    return t, gain, sub_set_dict['noMoreThanSetKey'], sub_set_dict['moreThanSetKey']


# 获得根据属性feature划分的熵增益
def get_gain(data_set, feature, n):
    entropy = get_entropy(data_set)
    feature_tags = data_set[:, feature]
    t, gain, no_more_than_set, more_than_set = caculate_t(data_set, feature_tags, entropy, n)
    return gain, t, no_more_than_set, more_than_set


# 计算某属性的增益率
def get_gain_ratio(gain, feature_tags):
    data_size = len(feature_tags)


# 从属性集中选择一个最好的属性，这里的方法基于的是C4.5中的方法
def select_feature(data_set, features, n):
    gain_list = list()
    t_list = list()
    no_more_than_set_list = list()
    more_than_set_list = list()
    for feature in features:
        gain, t, no_more_than_set, more_than_set = get_gain(data_set, feature, n)
        gain_list.append(gain)
        t_list.append(t)
        no_more_than_set_list.append(no_more_than_set)
        more_than_set_list.append(more_than_set)
    i = gain_list.index(max(gain_list))
    return features[i], t_list[i], no_more_than_set_list[i], more_than_set_list[i]


# 标记集合的数量最多的标记：1.属性集为空或取值一样 2.数据子集为空
def major_label(labels):
    tags = list(set(labels))
    tag_num = [sum([1 for i in labels if i == label]) for label in tags]
    k = tag_num.index(max(tag_num))
    return tags[k]


# 判断样本在当前属性集上取值是否完全相同/judge whether all the samples have the same values in the features
def has_same_values(data_set, features):
    for feature in features:
        if len(set(data_set[:, feature])) > 1:
            return False
    return True


# 由于所有属性皆为连续属性，每次在n个属性中随机取k个属性出来作为决策树的决策属性集
def get_random_features(k, n):
    features = list()
    for i in range(k):
        features.append(random.randint(0, n-1))
    return features


# 构建一棵决策树
def build_tree(data_set, features, k, n) -> dict:
    """

    :param data_set: 当前的数据集，DataFrame
    :param features: 选取的属性的索引列表
    :param k: 每次选取随机属性集合的大小
    :param n: 属性的总数
    :return:
    """
    labels = data_set[:, -1]
    if len(set(labels)) == 1:
        return {'label': labels[0]}
    if not len(features) or has_same_values(data_set, features):
        return {'label': major_label(labels)}
    #  根据t可以拆分两个子集 <t的子集和>t的子集,所以tree实际上是一棵二叉树，左分支<=t,右分支>t
    best_feature, t, no_more_than_set, more_than_set = select_feature(data_set, features, n)
    tree = {'feature': best_feature, 'T': t, 'child': {}}
    if len(no_more_than_set) == 0:
        tree['child']['leftChild'] = {'label': major_label(labels)}
    else:
        new_features = get_random_features(k, n)
        tree['child']['leftChild'] = build_tree(no_more_than_set, new_features, k, n)
    if len(more_than_set) == 0:
        tree['child']['rightchild'] = {'label': major_label(labels)}
    else:
        new_features = get_random_features(k, n)
        tree['child']['rightChild'] = build_tree(more_than_set, new_features, k, n)
    return tree


# 本次实现的是可以处理数据中同时有离散属性和连续属性的C4.5算法，每一次考虑如下方法：
# 首先选择信息增益大于平均水平的属性，注意连续属性的信息增益计算过程
# 然后对于这些属性考虑信息增益率，在C4.5中离散属性的增益率和连续属性的增益率定义是不同的
if __name__ == '__main__':
    pass
