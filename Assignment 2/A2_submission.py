import torch
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data.sampler import *
import torch.nn as nn
from pprint import pformat
from torch.utils.data import Subset

torch.multiprocessing.set_sharing_strategy('file_system')

# cost function for y


class LogisticRegression(nn.Module):
    def __init__(self, params):
        super(LogisticRegression, self).__init__()
        # your code here
        #linear using which ever size
        #28*28 for mnist,32*32*3 for cifar10 
        self.fc = nn.Linear(params['input_feature'], 10)
        
    def forward(self, x):
        # your code here
        # reshape x to make it into same dimension as fc
        # or so called flatening 
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def get_dataset(dataset_name):

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    transform_cifar10 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # set parameters mnist
    '''
    -> Parameters to feed in are 
        1. training dataset,
        2. validation
        3. test
    '''
    '''
    -> MNIST 
    '''
    batch_size_train = 120
    batch_size_test = 1000

    mnist_training = datasets.MNIST('./MNIST_dataset', train=True,
                                    download=True, transform=transform_mnist)

    # training set
    mnist_training_set = Subset(
        mnist_training, range(len(mnist_training)-12000))
    # validation set
    mnist_validation_set = Subset(mnist_training, range(
        len(mnist_training)-12000, len(mnist_training)))
    # test set
    mnist_test_set = datasets.MNIST('./MNIST_dataset', train=False,
                                    download=True, transform=transform_mnist)

    mnist_params = {
        'input_feature' : 28*28,
        'learning_rate': 1e-3,
        'weight_decay' : 1e-5,
        'epoch': 10,
    
        'optimizer':'Adam',
        'momentum':  0.95
    }
    '''
    -> CIFAR10 
    '''

    cifar10_training = datasets.CIFAR10(root='./CIFAR10_dataset', train=True,
                                        download=True, transform=transform_cifar10)
    # training set
    cifar10_training_set = Subset(
        cifar10_training, range(len(cifar10_training)-12000))
    # validation set
    cifar10_validation_set = Subset(
        cifar10_training, range(len(cifar10_training)-12000,len(cifar10_training)))
    # test set
    cifar10_test_set = datasets.CIFAR10(root='./CIFAR10_dataset', train=False,
                                        download=True, transform=transform_cifar10)

    # set parameters cifar10
    cifar10_params = {
        'input_feature' : 3*32*32,
        'learning_rate': 1e-3,
        'weight_decay' : 1e-5,
        'epoch': 5,

        'optimizer':'SGD',
        'momentum' : 0.95
    }

    
    #set distguishhing datas for each or the datasets
    if dataset_name == "MNIST":
        params = mnist_params
        train_dataloader = torch.utils.data.DataLoader(
            mnist_training_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
        valid_dataloader = torch.utils.data.DataLoader(
            mnist_validation_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(
            mnist_test_set, batch_size=batch_size_test, shuffle=True, num_workers=2)

    elif dataset_name == "CIFAR10":
        params = cifar10_params
        train_dataloader = torch.utils.data.DataLoader(
            cifar10_training_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
        valid_dataloader = torch.utils.data.DataLoader(
            cifar10_validation_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
        test_dataloader = torch.utils.data.DataLoader(
            cifar10_test_set, batch_size=batch_size_test, shuffle=True, num_workers=2)
        # your code here
    else:
        raise AssertionError(f'Invalid dataset: {dataset_name}')

    # your code here

    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloader
    }

    return dataloaders, params
'''

'''
#The test function tests accuracy of model after training
def test(model, test_dataloader, device, params):
    model.eval()
    test_predictions = []
    true_labels = []
    '''
    This block use 10000 testing dataset downloaded along with the training dataset.
    Tests accuracy of the current trained model by comparing the output of the trained model with "real target values" of the dataset
    It is to test the parameters.
    '''
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            # _ gets value, predicted gets indices
            _, predicted = torch.max(outputs, dim=1)

            predicted = predicted.cpu().numpy()
            target = target.cpu().numpy()

            test_predictions.append(predicted)
            true_labels.append(target)

    return torch.tensor(test_predictions), torch.tensor(true_labels)

#the validate function validates the accuracy of using the set hyperparameters after training
#it is set manually to validate training after 5 epoch
def validate(model, valid_dataloader, device, params):
    #evaluate the trained model 
    model.eval()
    mean_acc = 0
    total = 0
    correct = 0
    '''
    This block use 12000 dataset derived from trainingset to evalute the trained model.
    Tests accuracy of the current trained model by comparing the output of the trained model with "real target values" of the dataset
    It is to test the hyperparmeters.
    '''
    with torch.no_grad():
        for data,target in valid_dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _,predicted = torch.max(output,dim=1)
            total += target.size(0)
            correct += (predicted==target).sum().item()
    # your code here
    mean_acc = correct/total
    print(mean_acc)

    return mean_acc

#The training function trains model on datasets(mnist and cifar10) to improve accuracy of the model on the datasets.
def train(model, train_dataloader, valid_dataloader, device, 
            params,lr=None,momentum = None,weight_decay = None):
    # we use the Cross-entropy loss here
    if lr ==None:
        lr = params['learning_rate']
        momentum = params['momentum']
        weight_decay = params['weight_decay']

    model.train()
    if params['optimizer']=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=lr,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr = lr,
                                    momentum = momentum,
                                    weight_decay=weight_decay)
    mean_train_loss = nn.CrossEntropyLoss()
    
    #iterate over epoch
    for epoch in range(params['epoch']):
        running_loss = 0

        #iterate over the data and target of dataset
        for batch_idx, (data, target) in enumerate(train_dataloader):
            #use device (gpu)
            data = data.to(device)
            target = target.to(device)
            #reset grad 
            optimizer.zero_grad()
            output = model(data)
            #get loss with crossentropy loss equation in this term
            loss = mean_train_loss(output, target)
            #back prop
            loss.backward()
            #optimize by updating the parameters
            optimizer.step()
            # your code here
            #print(loss.item())
    
            running_loss += loss.item()
            
            if batch_idx % 119 == 0:    # print every 120 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch+1,batch_idx + 1, running_loss / 120))
                running_loss = 0.0
        #validate after every 5 epoch, print values of lr,momentum
        if (epoch+1)%5==0:
            print("Epoch is: %i, with Learning Rate: %f, and Momemtum: %f."%(epoch+1,lr,momentum))
            valid_acc = validate(model,valid_dataloader,device,params)
    return loss.item(),valid_acc

def tune_hyper_parameter(dataloaders, device, params):
    """update the following in your search"""
    best_optimizer = "Adam"
    best_hyperparams = {
        "regularizer": {
            'weight_decay':0
        
        },
        "Adam": {
            "accuracy": 0,
            "learning_rate": 0,
        },
        "SGD": {
            "accuracy": 0,
            "learning_rate": 0,
            "momentum": 0,
        }
    }
    """accumulate all validation accuracies you compute during hyper parameter search 
    for both optimizers"""
    validation_accuracy = []
    # your code here
    '''
    -> initialize random hyperparmater values 
    -> search for finding the optimal hyper-parameters
    '''
    learning_rate = torch.FloatTensor(5).uniform_(1e-3,1e-2)
    momentum = torch.FloatTensor(5).uniform_(0.9,1)
    weight_decay = torch.FloatTensor(5).uniform_(1e-5,1e-3)

    #load datas for trainingset and validationset
    train_dataloader = dataloaders['train']
    valid_dataloader = dataloaders['valid']

    #grab dictionary of each 
    params_regularizer = best_hyperparams['regularizer']
    params_Adam = best_hyperparams['Adam']
    params_SGD = best_hyperparams['SGD']


    '''
    Perform Grid Search for both optimizers:
    -> Use early termination as we get a good accuracy for hyperparameter values
    -> The boudary set is 0.39 because its the average value for accuracy of Logistic regression on test set.
    -> The accuracy for the validation set will be of a improvement when the parameters are right
    -> Hyperparameter search for Adam
    '''
    #set optimizer to use Adam in training 
    params['optimizer'] = 'Adam'
    for lr in learning_rate:
        for decay in  weight_decay:
            model = LogisticRegression(params).to(device)
            loss,accuracy = train(model, train_dataloader,valid_dataloader,device,
                                params,lr,0,decay)
            validation_accuracy.append(accuracy)
            
            #if accuracy is larger than previous, add hyperparameters and accuracy to dictionary
            if accuracy > params_Adam['accuracy']:
                params_Adam['accuracy'] = accuracy
                params_Adam['learning_rate'] = lr
                params_regularizer['weight_decay'] = decay
            
            print("Loss is: %f, Accuracy is %f."%(loss,accuracy))

            #break for when the accuracy updated is of 0.39 or larger
            if params_Adam['accuracy'] >=0.39:
                break
        if params_Adam['accuracy'] >=0.39:
                print("accuracy is bigger than expected, break")
                break
        
    print("\n==========================="
    "\nADAM accuracy: %f,lr: %f     "
    "\n"
    "\n"
    "\n==========================="%(params_Adam['accuracy'],params_Adam['learning_rate']))



 
                        

    '''
    -> Hyperparameter search for SGD
    '''
    #params_regularizer['optimizer'] = 'SGD'
    
    #if accuracy is larger than previous, add hyperparameters and accuracy to dictionary
    #set optimizer to SGD when training
    params['optimizer'] ='SGD'
    for lr in learning_rate:
        for mo in momentum:
            for decay in weight_decay:
                model = LogisticRegression(params).to(device)
                loss,accuracy = train(model, train_dataloader,valid_dataloader,device,
                                    params,lr,mo,decay)
                validation_accuracy.append(accuracy)
                print(loss,accuracy)
                if accuracy > params_SGD['accuracy']:
                    params_SGD['accuracy'] = accuracy
                    params_SGD['learning_rate'] = lr
                    params_SGD['momentum'] = mo
                    params_regularizer['weight_decay'] = decay
                
                #break for when the accuracy updated is of 0.39 or larger
                if params_SGD['accuracy'] >=0.39:
                    break
            if params_SGD['accuracy'] >=0.39:
                    break
        if params_SGD['accuracy'] >=0.39:
                    print("accuracy is bigger than expected, break")
                    break
        
    
    print("\n==========================="
    "\nSGD accuracy: %f,lr: %f,mo: %f  "
    "\n"
    "\n"
    "\n==========================="%(params_SGD['accuracy'],params_SGD['learning_rate'],params_SGD['momentum']))
    
   
    if params_SGD['accuracy']>params_Adam['accuracy']:
        best_optimizer = 'SGD'

    '''
    -> print result
    '''
    print("\nOptimal performance: Validation Accuracy: {:.3f}, "
          "with {:s} optimizer "
          "using hyper parameters:\n{:s} ".format(
              max(validation_accuracy),
              best_optimizer,
              pformat(best_hyperparams[best_optimizer])))

    print("\nOptimal regularization hyper parameters:\n{:s} ".format(
        pformat(best_hyperparams['regularizer'])))
