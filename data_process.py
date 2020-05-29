import random
import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.impute import SimpleImputer
data_dir = "./data/"
file_name = "breast-w.arff"


data = arff.loadarff(data_dir + file_name)
df = pd.DataFrame(data[0])
columns = df.columns
df_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df = df_frequent.fit_transform(df)
df = pd.DataFrame(df, columns=columns)
num = df.shape[0]


# 10-fold process
def k_fold_split(dataframe: pd.DataFrame, k, size):
    # build the fold index
    folds = []
    index = set(range(size))
    for i in range(k):
        if i == k-1:
            folds.append(list(index))
        else:
            temp = random.sample(list(index), int(size/k))
            folds.append(temp)
            index -= set(temp)

    # generate the k-fold training data and validation data
    fold_array = np.array(folds)
    for j in range(k):
        training_index = set(range(k)) - {j}
        vd = dataframe.iloc[np.array(folds[j])]
        training_array = fold_array[list(training_index)]
        training_line = np.concatenate(training_array, 0)
        td = dataframe.iloc[training_line]
        if not os.path.exists("./data/use/"):
            os.makedirs("./data/use/")
        td.loc[:, 'Class'] = td.loc[:, 'Class'].str.decode("utf-8")
        vd.loc[:, 'Class'] = vd.loc[:, 'Class'].str.decode("utf-8")
        td.to_csv(f"./data/use/train_{j}.csv", header=True, index=False)
        vd.to_csv(f"./data/use/val_{j}.csv", header=True, index=False)


if __name__ == '__main__':
    k_fold_split(df, 10, num)

