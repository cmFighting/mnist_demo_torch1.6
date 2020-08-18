import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 原图大小是28*28，经过两次卷积两次池化，并向上取整，最后的大小是
def load_data(data_path="E:/biye/gogogo/note_book/torch_note/data"):
    transform = transforms.Compose(
        # 这里只对其中的一个通道进行归一化的操作
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5,))])
    # 加载数据据
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                            transform=transform)
    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                         transform=transform)
    # 构建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    return trainloader, testloader


# 构造网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epochs=5, save_path='models/mnist_net.pth', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("程序执行设备：{}".format(device))
    trainloader, testloder = load_data()
    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                # print every 2000 mini-batches, 也就是每遍历8000个数据输出一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    # 保存模型
    torch.save(net.state_dict(), save_path)
    print('Finished Training And model saved in {}'.format(save_path))


if __name__ == '__main__':
    # load_data()
    train(device=torch.device("cpu"))
