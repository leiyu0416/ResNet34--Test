# ResNet34--Test

## What are the similarities between validation and testing sets? What are their respective purposes?

    Both the validation set and the test set are used to evaluate the model. 
    The validation set is used to further determine the hyperparameter in the model, while the test set is only used to evaluate the accuracy of the model.
    
    The validation set is used to select the sample set of optimal model hyperparameters, with the main purpose of selecting the model that performs best on the validation set.
    
    The test set is a dataset used to evaluate the generalization ability of the model after selecting the best performing model on the validation set.
 
 ## ResNet-34--Test
 ```import torch
import ResNet34

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 指定设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
model_path = "F:/实验/新建文件夹/dictionary/ResNet34/Hi05-25-23-42/ResNet34-epochs-5-model-val-acc-100.000-loss-0.001923.pt"
model = torch.load(model_path)
model.to(device)
model.eval()

# 指定数据集
test_dir = "data/test" 
test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor()
    ])
)

# 指定数据集加载器
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=4,  
    shuffle=False,
    num_workers=4  
)

# 定义损失函数
loss_function = torch.nn.CrossEntropyLoss()

# 定义一个完整的val模型，参考train.py中的val函数
def val():
    validating_loss = 0
    num_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # 测试中...
            data, target = data.to(device), target.to(device)
            output = model(data)
            validating_loss += loss_function(output, target).item()  # 累加 batch loss
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
            num_correct += pred.eq(target.view_as(pred)).sum().item()

            # 打印验证结果
            validating_loss /= len(test_loader)
            print('test >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
                validating_loss,
                num_correct,
                len(test_loader.dataset),
                100. * num_correct / len(test_loader.dataset))
            )
if __name__ == '__main__':
    val()
```

