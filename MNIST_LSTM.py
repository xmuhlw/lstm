
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam


# 定义超参数
BATCH_SIZE = 128
TIME_STEP = 28
INPUT_DIM = 28
HIDDEM_DIM = 100
LAYER_DIM = 1
OUTPUT_DIM = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1.准备数据集
def get_dataloader(train=True):
    dataset = MNIST(root="./data", train=train, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


# 2.构建模型
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # [time_step, batch_size, input]   [batch_size, time_step, input]
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        # [batch_size, time_step, hidden]
        return self.fc(out[:, -1, :])


model = LSTM_Model(input_dim=INPUT_DIM, hidden_dim=HIDDEM_DIM, layer_dim=LAYER_DIM, output_dim=OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 模型的加载
# if os.path.exists("./models/model.pkl"):
#     model.load_state_dict(torch.load("./models/model.pkl"))
#     optimizer.load_state_dict(torch.load("./models/optimizer.pkl"))


def train(epochs):
    for epoch in range(epochs):
        model.train()
        train_data_loader = get_dataloader(train=True)
        for idx, (images, labels) in enumerate(train_data_loader):
            # [batch_size, 1, 28, 28] --> [batch_size * 1, 28, 28]
            images, labels = images.view(-1, TIME_STEP, INPUT_DIM).to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(epoch, idx, loss.item())
                # 模型的保存
                torch.save(model.state_dict(), "./models/model.pkl")
                torch.save(optimizer.state_dict(), "./models/optimizer.pkl")


def test():
    correct = 0
    total = 0
    model.eval()
    test_data_loader = get_dataloader(train=False)
    for idx, (images, labels) in enumerate(test_data_loader):
        images, labels = images.view(-1, TIME_STEP, INPUT_DIM).to(device), labels.to(device)
        output = model(images)
        # output [batch_size, 10] target [batch_size]
        index = output.max(dim=-1)[-1]
        # print(index)
        total += labels.size(0)
        correct += (index == labels).float().sum()
        # break
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))


if __name__ == "__main__":
    train(3)
    test()
