from numpy import tile
import operator


def knn_predict(example, data_set, labels, k):
    """
    :param example: 单个测试样本（不包含标签）
    :param data_set: 训练集特征
    :param labels: 训练集标签
    :param k:
    :return:
    """
    m = data_set.shape[0]
    # 计算欧氏距离
    diff_mat = tile(example, (m, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    # 按distances中元素进行升序排序后得到的对应下标的列表
    sorted_dist_index = distances.argsort()
    # 选择距离最小的k个点，投票得出结果
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_index[i]]
        class_count[int(vote_label)] = class_count.get(vote_label, 0) + 1
    # 被分为正例的概率,注：这里所有二分类标签正类为0， 反例为1
    positive_prob = class_count.get(1, 0) / k
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0], positive_prob
