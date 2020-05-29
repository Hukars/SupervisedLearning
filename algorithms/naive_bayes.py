import numpy as np
import scipy.stats as st


# 统计每一个类别的频率
def get_class_probability(data_set, class_num):
    classes = data_set[:, -1]
    size = len(classes)
    count_dict = {}
    for e in classes:
        count_dict[e] = count_dict.get(e, 0) + 1
    prob_dict = count_dict.copy()
    laplace_dict = count_dict.copy()
    for c in count_dict.keys():
        prob_dict[c] = prob_dict[c] / size
        laplace_dict[c] = (laplace_dict[c] + 1) / (size + class_num)

    return count_dict, prob_dict, laplace_dict


# 估计类别生成属性的先验概率
def get_prior_probability(data_set, features_index, classes, is_discrete, class_dict):
    values = data_set[:, [features_index, -1]]
    class_value_dict = dict()
    for c in classes:
        class_value_dict[c] = dict()
    # 计数
    for value in values:
        class_value_dict[value[1]][value[0]] = class_value_dict[value[1]].get(value[0], 0) + 1
    if is_discrete:
        # 计算频率
        for i in class_value_dict.keys():
            for j in class_value_dict[i].keys():
                class_value_dict[i][j] = class_value_dict[i][j] / class_dict[i]
        return class_value_dict
    else:
        # 对于连续属性，不妨设其概率密度函数为高斯函数，计算均值和方差
        gaussian_dict = dict()
        for m in class_value_dict.keys():
            gaussian_dict[m] = []
            tmp_list = []
            tmp_sum = 0
            for n in class_value_dict[m].keys():
                tmp_sum += n * class_value_dict[m][n]
                tmp_list += [n] * class_value_dict[m][n]

            gaussian_dict[m].append(tmp_sum / sum(class_value_dict[m].values()))
            gaussian_dict[m].append(np.sqrt(np.var(tmp_list)))
        return gaussian_dict


def naive_bayes_training(training_data, class_num, feature_description):
    """
    :param training_data:
    :param class_num:
    :param feature_description: 关于每一维属性是离散还是连续的描述
    :return:
    """
    bayes_model = []
    count_dict, prob_dict, laplace_dict = get_class_probability(training_data, class_num)
    bayes_model += [count_dict, prob_dict, laplace_dict]
    for i in range(len(feature_description)):
        bayes_model.append(get_prior_probability(training_data, i, [0, 1], feature_description[i], count_dict))
    return bayes_model


# 进行朴素贝叶斯预测过程，对于离散属性，需要考虑使用拉普拉斯平滑技术
def naive_bayes_predict(examples, feature_parameters, classes, classes_prob, features_tag):
    """
    基于模型预测结果
    :param examples: 输入的预测样本，numpy array
    :param feature_parameters: 模型参数，每一维的先验概率
    :param classes: 以二分类数据集为例：[0, 1]
    :param classes_prob: [count_dict, prob_dict, laplace_dict]
    :param features_tag:  [feature_description, feature_values_num]
    :return: 类别预测和属于正例的概率
    """
    m, n = examples.shape
    predict_results = []
    prob_results = []
    for example in examples:
        tmp = []
        laplace = False
        for c in classes:
            prob = classes_prob[1][c]
            for f in range(len(features_tag[0])):
                parameters = feature_parameters[f]
                if features_tag[0][f]:
                    # 判断是否为0概率, 0概率用拉普拉斯平滑处理
                    if example[f] in parameters[c].keys():
                        prob *= parameters[c][example[f]]
                    else:
                        laplace = True
                        prob *= 1 / (classes_prob[0][c] + features_tag[1][f])
                else:
                    prob *= st.norm(example[f], parameters[c][0], parameters[c][1])
            if laplace:
                prob = prob / classes_prob[1][c] * classes_prob[2][c]
            tmp.append(prob)
        predict_results.append(np.argmax(tmp))
        prob_results.append(tmp[-1])

    return np.expand_dims(np.array(predict_results), 1), np.expand_dims(np.array(prob_results), 1)






