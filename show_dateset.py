import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')


def imshow(img):
    # 输入结果
    # 注意，这里传入的是灰度图像，而不是二值图像
    print(img.shape)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# TODO 请将这里的路径切换成你本地的路径
def show_some_data(data_path="E:/biye/gogogo/note_book/torch_note/data"):
    # 输出图像的函数
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize((0.5, ), (0.5, ))])
    # 加载数据据
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                          transform=transform)
    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                         transform=transform)
    # print(trainset.size())
    # 构建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # 随机获取训练图片
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)

    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印图片标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    show_some_data()