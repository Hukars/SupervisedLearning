import torch
import time
import numpy as np
import pandas as pd
from sklearn import metrics
from algorithms.neural_network import neural_network_training
from algorithms.knn import knn_predict
from algorithms.svm_api import svm_training
from algorithms.naive_bayes import naive_bayes_training, naive_bayes_predict

data_base_dir = "./data/use/"


def load_data(training_data_dir, testing_data_dir, class_list):
    training_data_pd = pd.read_csv(training_data_dir)
    testing_data_pd = pd.read_csv(testing_data_dir)
    training_data_pd["Class"] = training_data_pd['Class'].apply(lambda x: class_list.index(x))
    testing_data_pd["Class"] = testing_data_pd['Class'].apply(lambda y: class_list.index(y))
    return training_data_pd.to_numpy(), testing_data_pd.to_numpy()


def bootstrap(data_set):
    m, n = data_set.shape
    result = np.empty([m, n])
    for i in range(m):
        result[i] = data_set[np.random.randint(0, m)]
    return result


def bagging(training_data, testing_data, T, algorithm_number, class_num, features_tag):
    model_list = []
    for t in range(T):
        # print(f"进行第{t+1}个基模型的训练")
        cur_training_data = bootstrap(training_data)
        if algorithm_number == 1:
            cur_model = naive_bayes_training(training_data, class_num, features_tag[0])
            model_list.append(cur_model)
        elif algorithm_number == 2:
            cur_model = svm_training(training_data)
            model_list.append(cur_model)
        elif algorithm_number == 3:
            cur_model = neural_network_training(cur_training_data, class_num)
            model_list.append(cur_model)
        elif algorithm_number == 4:
            model_list.append(cur_training_data)

    vote_recording = []
    positive_prob_recording = []
    testing_y = testing_data[:, -1]
    testing_x = testing_data[:, :-1]
    if algorithm_number == 3:
        testing_x = torch.from_numpy(testing_x)
        for model in model_list:
            predicts = model(testing_x.float())
            predict_y = predicts.argmax(1, keepdim=True)
            positive_prob = predicts[:, 1].unsqueeze(1)
            vote_recording.append(predict_y)
            positive_prob_recording.append(positive_prob)
        vote_recording = torch.cat(vote_recording, 1).int().numpy()
        positive_prob_recording = torch.cat(positive_prob_recording, 1)
        positive_prob_recording = positive_prob_recording.mean(1).detach().numpy()
    else:
        if algorithm_number == 1:
            for model in model_list:
                predicts, positive_prob = naive_bayes_predict(testing_x, model[3:], np.arange(class_num),
                                                              model[0:3], features_tag)
                vote_recording.append(predicts)
                positive_prob_recording.append(positive_prob)
        elif algorithm_number == 2:
            for model in model_list:
                predicts = model.predict(testing_x)
                predicts = np.expand_dims(predicts, 1)
                vote_recording.append(predicts)
                positive_prob = np.exp(model.predict_log_proba(testing_x))
                positive_prob_recording.append(positive_prob)
        elif algorithm_number == 3:
            testing_x = torch.from_numpy(testing_x)
            for model in model_list:
                predicts = model(testing_x.float())
                predict_y = predicts.argmax(1, keepdim=True)
                positive_prob = predicts[:, 1].unsqueeze(1)
                vote_recording.append(predict_y)
                positive_prob_recording.append(positive_prob)
        if algorithm_number == 4:
            m, _ = testing_x.shape
            for model in model_list:
                result_array, prob_array = np.empty([m, 1]), np.empty([m, 1])
                for i in range(m):
                    knn_result, knn_positive_prob = knn_predict(testing_x[i], model[:, :-1], model[:, -1], 6)
                    result_array[i] = knn_result
                    prob_array[i] = knn_positive_prob
                vote_recording.append(result_array)
                positive_prob_recording.append(prob_array)
        vote_recording = np.concatenate(vote_recording, 1).astype('int64')
        positive_prob_recording = np.concatenate(positive_prob_recording, 1)
        positive_prob_recording = positive_prob_recording.mean(1)

    # 根据预测结果计算acc，auc
    vote_result = []
    for row in vote_recording:
        vote_result.append(np.argmax(np.bincount(row)))
    vote_result = np.array(vote_result)
    accuracy = metrics.accuracy_score(testing_y, vote_result)
    auc = metrics.roc_auc_score(testing_y, positive_prob_recording)
    return accuracy, auc


# 目前针对的主要是二分类问题
if __name__ == '__main__':
    val_time = 10
    T = 5
    code = [(1, "Naive Bayes"), (2, "SVM"), (3, "Neural Network"), (4, "kNN")]
    class_num = 2
    class_list = ['malignant', 'benign']
    # 参见breast-w.arff数据集
    features_tag = [[True] * 9, [10]*9]
    # 对于每一种算法
    for algorithm_code, algorithm in code:
        start = time.time()
        print(f"现在开始{algorithm}-bagging算法十折交叉验证过程")
        total_acc, total_auc = 0.0, 0.0
        for cross in range(val_time):
            print(f"进行第{cross + 1}次交叉验证训练")
            cross_training_data_dir = data_base_dir + f"train_{cross}.csv"
            cross_testing_data_dir = data_base_dir + f"val_{cross}.csv"
            cross_training_data, cross_testing_data = load_data(cross_training_data_dir, cross_testing_data_dir,
                                                                class_list)

            acc, auc = bagging(cross_training_data, cross_testing_data, T, algorithm_code, class_num, features_tag)
            total_acc += acc
            total_auc += auc
        time_cost = time.time() - start
        print(f"{algorithm}-bagging算法十折交叉训练结果：算法用时:{time_cost}s, 平均准确率:{total_acc / val_time}, 平均auc:{total_auc / val_time}")
        break
