import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train_based_torchvision import Net


# 相互转换 https://blog.csdn.net/qq_37828488/article/details/96628988?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param
def imshow(img, name):
    name = name.numpy()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    # 图片转换这里存在一些小问题，之后再解决
    npimg = npimg * 255
    npimg.astype(np.int)
    # 通过asarray就能将图片矫正过来
    img = cv2.cvtColor(np.asarray(npimg), cv2.COLOR_RGB2BGR)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(str(name) + '.jpg', img)


def get_some_imgs(data_path="E:/biye/gogogo/note_book/torch_note/data"):
    transform = transforms.Compose(
        # 这里只对其中的一个通道进行归一化的操作
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    testset = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    img = images[3]
    label = labels[3]

    imshow(img, label)


def single_img_predict(model_path='models/mnist_net.pth', img_path="imgs/0.jpg"):
    # 加载网络
    net = Net()
    net.load_state_dict(torch.load(model_path))

    transform = transforms.Compose(
        # 这里只对其中的一个通道进行归一化的操作
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    img = Image.open(img_path)
    gray_img = img.convert('L')
    img_torch = transform(gray_img)
    img_torch = img_torch.view(-1, 1, 28, 28)

    # 开始预测
    outputs = net(img_torch)
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    print("{}的预测结果是:{}".format(img_path, predicted[0].numpy()))
    cv2.imshow("current_img", cv2.imread(img_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 转化一些图片出来
    # get_some_imgs()
    # 进行测试
    single_img_predict(img_path="imgs/7.jpg")
