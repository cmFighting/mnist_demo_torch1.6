from train_based_torchvision import load_data, Net
import torch


def test_test_dataset(model_path='models/mnist_net.pth'):
    # 加载模型
    net = Net()
    net.load_state_dict(torch.load(model_path))
    trainloader, testloader = load_data()
    # 测试全部的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def test_test_dataset_by_classes(model_path='models/mnist_net.pth'):
    # 加载模型
    net = Net()
    net.load_state_dict(torch.load(model_path))
    trainloader, testloader = load_data()
    classes = [str(i) for i in range(10)]
    # 测试每一类的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    print('模型在整个数据集上的表现：')
    test_test_dataset()
    print('模型在每一类上的表现：')
    test_test_dataset_by_classes()