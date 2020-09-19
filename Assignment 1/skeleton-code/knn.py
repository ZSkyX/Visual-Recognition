import torch
def knn(x_train, y_train, x_test, n_classes, device):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 1000 x 784 testing images
    n_classes: no. of classes in the classification task (digit 0-9)
    device: pytorch device on which to run the code
    return: predicted y_test which is a 1000-sized vector
    """
    """
    train:  60000 
    test:  1000 
    Q: find k
    """
    
    """
    covert numpy array to tensor
    """
    
    k = 4
    
    #convert to tensor
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    
    y_test = []
    
    for test in x_test:
      #distance =torch.argsort(torch.sqrt(torch.sum(torch.pw(torch.sub(test,x_train),2),axis=1)))[:k]
      #L2 Norm
      distance = torch.argsort(torch.norm(test-x_train,dim=1))[:k]
      #get from y_train
      temp = [y_train[i] for i in distance]
      #convert to tensor to use tensor.mode
      temp = torch.tensor(temp)
      #prediction of y
      y_test.append(torch.mode(temp)[0])
    

    return y_test
    
    
   