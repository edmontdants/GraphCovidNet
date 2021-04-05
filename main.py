import time
import random
import sys
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from statistics import mean, pstdev
import torch.nn.functional as F

from model import *
from dataloader import *

import seaborn as sns

from sklearn.metrics import f1_score, precision_score, recall_score,roc_curve,auc,classification_report,confusion_matrix

from tensorflow.keras.utils import to_categorical

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('cuda', action='store_true', default=False,
                    help='Use CUDA for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay for optimizer.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_split', type=float, default=0.8,
                    help='Ratio of train split from entire dataset.Rest goes to test set')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for loading mini batches of data')
parser.add_argument('--dataset_name', type=str, default='Kaggle_Pretwitt',
                    help='Dataset name')

args = parser.parse_args()
if args.cuda:
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.cuda.manual_seed(args.seed)
	else:
		print("Sorry no gpu found!!")
		device=torch.device('cpu')
		print("Running model on cpu")
else:
	device=torch.device('cpu')

#Setting seed to reproduce results
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = GraphDataset(root='/content/drive/My Drive/GraphTrain_1/dataset/', name=args.dataset_name, use_node_attr=True)
data_size = len(dataset)

#checking some of the data attributes comment out these lines if not needed to check
print("*"*10)
print(data_size)
print(dataset.num_features)
print(args.hidden)
print(args.dropout)
print(dataset.num_classes)
print("*"*10)

n_classes=dataset.num_classes
class_list=[]
for i in range(0,n_classes):
    class_list.append(i)
    

#printing confusion matrix
def show_confusion_matrix(validations, predictions):
  
    LABELS=["COVID","NON-COVID","PNEUMONIA"]
    matrix = confusion_matrix(validations, predictions)
    print(matrix)
    

#applying k-fold cross validation
def crossvalid(dataset=None,k_fold=5):
    
    global precision,recall,f1
    
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction) 
    index=0
    test_accs=[]
    train_time=0
    test_time=0
    for i in range(k_fold):
        if(i==k_fold-1):
            print("Running for {} fold".format(index+1))
            index=index+1
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_size
            
            train_left_indices = list(range(trll,trlr))
            train_right_indices = list(range(trrl,trrr))
            
            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall,valr))
            
            train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
            val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
            
            print(len(train_set),len(val_set))
            
            train_loader = DataLoader(train_set, batch_size=1,
                                              shuffle=True)
            val_loader = DataLoader(val_set, batch_size=1,
                                              shuffle=True)
                                              
            
            model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
            opt = optim.Adam(model.parameters(), lr=0.001)
            
            loss_values=[]
            accuracy_values=[]
            
            train_start=time.time()  
            for epoch in range(1):
              total_loss = 0
              model.train()
              for batch in train_loader:
                
                opt.zero_grad()
                embedding, pred, soft= model(batch)
                label = batch.y
                if task == 'node':
                    pred = pred[batch.train_mask]
                    label = label[batch.train_mask]
                loss = model.loss(pred, label)
                loss.backward()
                opt.step()
                total_loss += loss.item() * batch.num_graphs
              total_loss /= len(train_loader.dataset)
              writer.add_scalar("loss", total_loss, epoch)
    
              if epoch % 1 == 0:
                train_acc = test(train_loader, model)
                print("Epoch {}. Train Loss: {:.4f}. Train accuracy: {:.4f}".format(
                    epoch, total_loss, train_acc))
                    
                loss_values.append(total_loss)
                accuracy_values.append(train_acc)
                
                writer.add_scalar("train accuracy", train_acc, epoch)
            
            train_end=time.time()
            print("Time taken for training: ",train_end-train_start)
            train_time+=(train_end-train_start)
            
            test_start=time.time()
            test_acc = test(val_loader, model, True)
            test_end=time.time()
            print("Test accuracy: {:.4f}".format(test_acc))
            print("Time taken for testing: ",test_end-test_start)
            test_time+=(test_end-test_start)
            
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(111)
            ax.set_title('Training loss and accuracy Vs Epoch')
            plt.plot(loss_values, color='red',label='Loss')
            plt.plot(accuracy_values, color='blue',label='Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss and Accuracy')
            ax.legend(loc='best')
            plt.savefig(f"{k_fold}-fold_"+args.dataset_name+"_"+str(i+1)+".png")
            
            test_accs.append(test_acc)
                
    precision/=k_fold
    recall/=k_fold
    f1/=k_fold
    avg_test=sum(test_accs)/len(test_accs)
    train_time/=k_fold
    test_time/=k_fold
    print("------------")
    print("{} fold test accuracy {:.4f}, precision {:.4f}, recall {:.4f}, F1-score {:.4f}".format(k_fold,avg_test,precision,recall,f1))
    print("Average training time {:.3f}, average testing time {:.3f}".format(train_time,test_time))    


def train(dataset, task, writer):
    crossvalid(dataset)
    
#function for calculating acc for different train-ratio
def eval_train_ratio(dataset, task, writer):
    
    train_ratios=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    test_accuracies=[]
    train_accuracies=[]
    for tr in train_ratios:
        if task == 'graph':
            data_size = len(dataset)
            train_loader = DataLoader(dataset[:int(data_size * tr)], batch_size=1, shuffle=True)
            test_loader = DataLoader(dataset[int(data_size * tr):], batch_size=1, shuffle=True)
        else:
            test_loader = train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
        print("Train ratio: {:.1f}. No of training graphs: {}. No of testing graphs: {}".format(tr,len(train_loader),len(test_loader)))
        
        model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
        opt = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
          total_loss = 0
          model.train()
          for batch in train_loader:
            
            opt.zero_grad()
            embedding, pred ,_= model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
          total_loss /= len(train_loader.dataset)
          
        
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model, True)
        print("Test accuracy: {:.4f}".format(test_acc))
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)
        
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_title('Training and testing accuracy Vs Train Ratio')
    plt.plot(train_ratios, train_accuracies, color='red', marker= 'o',label='Train Acc')
    plt.plot(train_ratios, test_accuracies, color='blue',marker= 'o', label='Test Acc')
    ax.set_xlabel('Train Ratio')
    ax.set_ylabel('Training and testing accuracy')
    ax.legend(loc='best')
    #plt.show()
    plt.savefig(f"iterative_training_"+args.dataset_name+".png")
    
def test(loader, model, is_test=False,is_validation=False):
    global precision,recall,f1
    global class_list,n_classes
    global cnt
    model.eval()
    
    correct = 0
    glabel=[]
    glabel1=[]
    gpred=[]
    gscore=[]
    for data in loader:
        with torch.no_grad():
            emb, pred,soft = model(data)
            var=soft.numpy()[0]
            pred = pred.argmax(dim=1)
            label = data.y
            
            
            if(is_test): 
                glabel.append(label.numpy()[0])
                glabel1.append(label.numpy())   
                gpred.append(pred.numpy()[0])
                gscore.append(var)
            

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    
    if(is_test):
        glabel=np.array(glabel)
        glabel1=np.array(glabel1)
        gpred=np.array(gpred)
        gscore=np.array(gscore)
        enlabel=to_categorical(glabel1,n_classes)
        
        
        
        p=precision_score(glabel, gpred, average="micro")
        r=recall_score(glabel, gpred, average="micro")
        f=f1_score(glabel, gpred, average="micro")
        
        print('F1: {}'.format(f))
        print('Precision: {}'.format(p))
        print('Recall: {}'.format(r))
        precision+=p
        recall+=r
        f1+=f
        
        print("\n...confusion matrix and classification report....\n")
        show_confusion_matrix(glabel,gpred)
        print(classification_report(glabel,gpred))

        
        #generate roc curve
        tpr=dict()
        fpr=dict()
        roc_auc=dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _= roc_curve(enlabel[:, i], gscore[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_title('ROC curve for all classes')
        colors=['red','blue','green','yellow','purple','orange']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i],lw=2,label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))    
        ax.set_xlabel('False postive rate')
        ax.set_ylabel('True postive rate')
        ax.legend(loc='best')
        plt.savefig(f"roc_"+args.dataset_name+"_"+str(cnt)+".png")
        cnt+=1
        
    return correct / total
dataset = dataset.shuffle()
task = 'graph'
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

#globals 
precision=0
recall=0
f1=0

cnt=1

train(dataset, task, writer)
eval_train_ratio(dataset, task, writer)




