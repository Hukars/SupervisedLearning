from sklearn.svm import SVC


def svm_training(training_data):
    training_x = training_data[:, :-1]
    training_y = training_data[:, -1]
    clf = SVC(gamma='auto', probability=True)
    clf.fit(training_x, training_y)
    return clf