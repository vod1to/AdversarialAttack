from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torcheval.metrics import MulticlassAUPRC
from torcheval.metrics import MulticlassF1Score
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as f
import medmnist
from ignite.metrics import Precision
from ignite.metrics import Recall
from ignite.metrics import RocCurve
from ignite.metrics import PrecisionRecallCurve
import matplotlib.pyplot as plt
from medmnist import INFO, Evaluator
from ignite.engine import Engine
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch.nn.functional as f
import medmnist
from medmnist import INFO, Evaluator


print(f"MedMNIST v {medmnist.__version__} @ {medmnist.HOMEPAGE}")
data_flag = 'breastmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

BATCH_SIZE = 128 
NUM_EPOCHS = 10
lr = 0.001

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    RandomHorizontalFlip(),
    RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = DataClass(
    split='train', transform=data_transform, download=download)
test_dataset = DataClass(
    split='test', transform=test_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

train_dataset.montage(length=20)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, n_classes)
)


auprc_metric = MulticlassAUPRC(num_classes=n_classes)
precision = Precision()
recall = Recall()
f1_metric = MulticlassF1Score(
    num_classes=n_classes, average='micro')


if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

def eval_step(engine, batch):
    return batch


RocCurve_evaluator = Engine(eval_step)
PrecisionRecallCurve_evaluator = Engine(eval_step)


def output_transform(output):
    y_pred, y_true = output
    return y_pred[:, 1], y_true

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    model.train()
    auprc_metric.reset()

    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
    scheduler.step()


def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = torch.sigmoid(outputs)
                auprc_metric.update(outputs, targets)
                f1_metric.update(outputs, targets)
                precision.update((outputs, targets))
                recall.update((outputs, targets))
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                auprc_metric.update(outputs, targets)
                f1_metric.update(outputs, targets)
                precision.update((outputs, targets))
                recall.update((outputs, targets))
                targets = targets.float().resize_(len(targets), 1)
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_score_evaluate = y_score.detach().numpy()
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score_evaluate)
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
        return y_true, y_score


print('==> Evaluating ...')

y_true_train, y_score_train = test('train')
y_true_test, y_score_test = test('test')
auprc_score = auprc_metric.compute()
f1_score = f1_metric.compute()
precision_score = precision.compute()
recall_score = recall.compute()


print('AUPR score: ' + str(auprc_score.item()))
print('precision score: ' + str(precision_score))
print('Recall score: ' + str(recall_score))
print('F1 score: ' + str(f1_score.item()))

global_y_true = torch.cat((y_true_train, y_true_test), 0)
global_y_score = torch.cat((y_score_train, y_score_test), 0)
roc_auc = RocCurve(output_transform=output_transform)
roc_auc.attach(RocCurve_evaluator, 'roc_auc')
state = RocCurve_evaluator.run(
    [[global_y_score, global_y_true]])

fpr, tpr, roc_thresholds = state.metrics['roc_auc']
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Test and Val data')
plt.legend()
plt.show()

prec_recall_curve = PrecisionRecallCurve(
    output_transform=output_transform)
prec_recall_curve.attach(
    PrecisionRecallCurve_evaluator, 'prec_recall_curve')
state = PrecisionRecallCurve_evaluator.run([[global_y_score, global_y_true]])
precision, recall, pr_thresholds = state.metrics['prec_recall_curve']
plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Test and Val Data')
plt.grid(True)
plt.tight_layout()
plt.show()
