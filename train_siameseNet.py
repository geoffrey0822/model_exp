import os,sys,argparse
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from dataset.CrossedLabelDataset import CrossedLabelDataset
from torchvision.models.alexnet import alexnet
from torchvision.models.googlenet import googlenet
from torchvision.models.resnet import resnet18
from torchvision.models.densenet import densenet121
from torch.autograd import Variable
from models.siameseModel import SiameseModel

supported_models=['alexnet', 'googlenet', 'resnet', 'densenet']

parser = argparse.ArgumentParser(description='Trainer for siamese')
parser.add_argument('--model_name', dest='model_name', default='alexnet',help='[alex,googlenet,resnet,densenet')
parser.add_argument('--data_path', dest='data_path', default=None)
parser.add_argument('--train_file', dest='train_file',default=None)
parser.add_argument('--val_file', dest='val_file',default=None)
parser.add_argument('--epoch', dest='epoch',type=int,default=30)
parser.add_argument('--shuffle', dest='shuffle',type=int,default=1)
args = parser.parse_args()

model_name = args.model_name.lower()
data_path = args.data_path
train_file = args.train_file
val_file = args.val_file
epoch = args.epoch
needShuffle = False

if model_name is None or model_name not in supported_models:
    print('please provide the valid model name.')
    exit(-1)

if data_path is None:
    print('please provide the root path for dataset')
    exit(-1)

if train_file is None:
    print('please provide the training file')
    exit(-1)

if epoch<1:
    print('invalid epoch value')
    exit(-1)

if args.shuffle==1:
    needShuffle=True

if not os.path.isdir(data_path):
    print('%s is not a directory'%data_path)
    exit(-1)

if not os.path.isfile(train_file):
    print('%s is not found'%train_file)
    exit(-1)

training_set = CrossedLabelDataset(label_file=train_file, root_dir=data_path)
print('training set has been loaded.')
val_set = None
if val_file is not None and os.path.isfile(val_file):
    val_set = CrossedLabelDataset(label_file=val_file, root_dir=data_path)
    print('validation set has been loaded.')

nn_backbond = None
if model_name=='alexnet':
    nn_backbond = alexnet().feature.cuda()
elif model_name=='resnet':
    nn_backbond = resnet18().feature.cuda()
elif model_name=='googlenet':
    nn_backbond = googlenet().feature.cuda()
elif model_name=='densenet':
    nn_backbond = densenet121().feature.cuda()
else:
    pass

print('%s has been loaded.'%nn_backbond.__class__.__name__)
nn_model = SiameseModel(nn_backbond, 4*4*256, 3072)
print('%s has been loaded.'%nn_model.__class__.__name__)
train_dl = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True)

loss_function = torch.nn.TripletMarginLoss()
optimizer = optim.SGD(nn_model.parameters(), lr=0.001)

training_set.printMe()

for iepoch in range(epoch):
    train_loader_iter = iter(train_dl)
    print('training %d epoch'%iepoch)
    epoch_loss = 0
    for batch_idx, (ims, true_ims, false_ims) in enumerate(train_loader_iter):
        nn_model.zero_grad()
        ims, true_ims, false_ims=Variable(ims.float().cuda()), Variable(true_ims.float().cuda()), Variable(false_ims.float().cuda())
        
        output1 = nn_model.feature(ims)
        output2 = nn_model.feature(true_ims)
        output3 = nn_model.feature(false_ims)
        output = loss_function(output1, output3, output2)
        output.backward()
        optimizer.step()
        epoch_loss = loss_function.item()
        print('loss:%f'%epoch_loss)


print('Done')