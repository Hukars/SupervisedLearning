import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader


class PredictNet(nn.Module):
    def __init__(self, input_dim, class_num, hidden_size):
        super(PredictNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, class_num),
            nn.Softmax()
        )

        for m in self.model:
            if type(m) == nn.Linear:
                m.bias.data.fill_(0.0)
                size_in = m.in_features
                m.weight.data.normal_(0.0, 1/size_in)

    def forward(self, x):
        return self.model(x)


class MyDataSet(Dataset):
    def __init__(self, training_data):
        self.training_x = torch.from_numpy(training_data[:, :-1]).float()
        self.training_y = torch.from_numpy(training_data[:, -1]).float().unsqueeze(1)
        self.size = training_data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.training_x[item], self.training_y[item]


def neural_network_training(training_data, classes_num):
    m, n = training_data.shape
    data_set = MyDataSet(training_data)
    data_loader = DataLoader(dataset=data_set, batch_size=256, shuffle=True)
    epoch = 200
    learning_rate = 1e-4
    hidden_size = 128
    predict_net = PredictNet(n-1, classes_num, hidden_size)
    optimizer = opt.Adam(predict_net.parameters(), lr=learning_rate)
    loss_func = nn.BCELoss()

    for e in range(epoch):
        print(f"Start epoch{e+1}")
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            predicts = predict_net(inputs)
            loss = loss_func(predicts[:, 1].unsqueeze(1), labels)
            print(f"Batch{i+1},batch_loss:{loss}")
            loss.backward()
            optimizer.step()
        print(f"Finish epoch{e+1}")

    return predict_net


