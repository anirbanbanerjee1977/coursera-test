import torch
import argparse
from torch import nn
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torch.autograd import Variable
import json
from collections import OrderedDict
import os.path


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
    
   Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default = 'flowers',
                    help='root directory for training,validation and test dataset image files')
    parser.add_argument('--arch', type=str, default = 'vgg',
                    help='selected CNN model')
    parser.add_argument('--learning_rate', type=float, default = 0.0001,
                    help='learning rate')
    parser.add_argument('--hidden_units', type=int, default = 512,
                    help='input to hidden_units')
    parser.add_argument('--epochs', type=int, default = 3,
                    help='number of epochs')
    parser.add_argument('--gpu', type=str, default = 'gpu',
                    help='processor type')
    parser.add_argument('--save_dir', type=str, default = '/home/workspace/aipnd-project',
                    help='directory where the model checkpoint saved')
    
    
    parse_args = parser.parse_args()
    return parse_args

def get_prediction_inputs():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
    
   Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default = 'vgg',
                    help='selected CNN model')
    parser.add_argument('--learning_rate', type=float, default = 0.0001,
                    help='learning rate')
    parser.add_argument('--gpu', type=str, default = 'gpu',
                    help='processor type')
    parser.add_argument('--path_to_image', type=str, default = '/home/workspace/aipnd-project/flowers/train/1/image_06734.jpg',
                    help='path to the image file')
    parser.add_argument('--category_names', type=str, default = 'cat_to_name.json',
                    help='file name containing image category')
    parser.add_argument('--top_k', type=int, default = 3,
                    help='top k probablities of image predictions')
    parser.add_argument('--checkpoint', type=str, default = '/home/workspace/aipnd-project',
                    help='directory from where the model checkpoint will be loaded')
    
    parse_args = parser.parse_args()
    return parse_args


def setup_directories(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    return train_dir,valid_dir,test_dir

def setup_transforms():
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    return training_transforms,validation_transforms,testing_transforms

def load_datasets(train_dir,valid_dir,test_dir,training_transforms,validation_transforms,testing_transforms):
    training_datasets = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir, transform=testing_transforms)
    
    return training_datasets,validation_datasets,testing_datasets
    
def setup_dataloaders(training_datasets,validation_datasets,testing_datasets):
    training_loader = torch.utils.data.DataLoader(training_datasets, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    test_loader = torch.utils.data.DataLoader(testing_datasets, batch_size=32)
    return training_loader,validation_loader,test_loader
    
def map_labels(category_name):
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def buildNN(arch,hidden_units):
    #Step 1 : Load a pre-trained network
    if arch == 'vgg':
        model = models.vgg19(pretrained=True)
    elif arch == 'resnet':
        model = models.resnet101(pretrained=True)
        
    # Freeze the parameters of model so that losses doesn't back propagate

    for param in model.parameters():
        model.requires_grad = False

    #Step 2 : Define a new,untrained feed-forward network as a classifier, using ReLU activations and dropout
    if arch == 'vgg':
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 2048)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(2048, 512)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif arch == 'resnet':
        classifier = nn.Sequential(OrderedDict( [ 
            ('fc1', nn.Linear(1024, 500)), 
            ('relu', nn.ReLU()), 
            ('fc2', nn.Linear(500, 102)), 
            ('output', nn.LogSoftmax(dim=1)) ]))
        
    model.classifier = classifier
    return model

def trainNN(model,training_loader,validation_loader,train_epochs,learning_rate,gpu):
    #Train the classifier layers using backpropagation using the pre-trained network to get the features
    #Track the loss and accuracy on the validation set to determine the best hyperparameters

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    epochs = train_epochs
    print_every = 40
    steps = 0
    if gpu == 'gpu':
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
    
        for ii, (inputs, labels) in enumerate(training_loader):
            steps += 1
            if gpu == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')     
            
            optimizer.zero_grad()
        
            # Forward and backward passes
        
            outputs = model.forward(inputs)
            if gpu == 'gpu':
                loss = criterion(outputs, labels).cuda()
            else:
                loss = criterion(outputs, labels)
        
            loss.backward()
        
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                ## change
                model.eval()
                accuracy = 0
                val_loss = 0
                for ival, (val_input, val_label) in enumerate(validation_loader):
                    if gpu == 'gpu':
                        val_input, val_label = val_input.to('cuda'), val_label.to('cuda')    
                    val_output = model.forward(val_input)
                    loss_fn = nn.CrossEntropyLoss()
                    val_loss += loss_fn(val_output, val_label).data[0]

                    ps = torch.exp(val_output).data
                
                    equality = (val_label.data == ps.max(1)[1])
                
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    ## change
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss : {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss : {:.3f}.. ".format(val_loss/len(validation_loader)),
                        "Validation Accuracy : {:.3f}.. ".format(accuracy))

                running_loss = 0
    return model,optimizer
            
def testNN(model,validation_loader,test_loader,gpu):
    # Do validation on the validation set
    correct_validation = 0
    total_validation = 0
    with torch.no_grad():
        for data_validation in validation_loader:
            images_validation, labels_validation = data_validation
            if gpu == 'gpu':
                images_validation, labels_validation = images_validation.cuda(), labels_validation.cuda()
            outputs_validation = model(images_validation)
            _, predicted_validation = torch.max(outputs_validation.data, 1)
            total_validation += labels_validation.size(0)
            correct_validation += (predicted_validation == labels_validation).sum().item()

    print('Accuracy of the network on the 10000 validation images: %d %%' % (100 * correct_validation / total_validation))


# Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if gpu == 'gpu':
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
def saveCheckPoint(model,training_datasets,epochs,optimizer,save_dir):
    #Save the checkpoint 
    model.class_to_idx = training_datasets.class_to_idx
    idx_to_class = {}
    for name, label in model.class_to_idx.items():
        idx_to_class[label] = name
        print_every = 40
        checkpoint = {'epochs': epochs,
              'number of hidden units used':2,
              'batch size':print_every,
                'optimizer_state': optimizer.state_dict(),
              'state_dict': model.state_dict(),
                     'classifier': model.classifier,
                     'class_to_idx' : model.class_to_idx}
    torch.save(checkpoint, save_dir + '/checkpoint.pth')


def load_checkpoint(filepath,learning_rate,arch):
    #function that loads a checkpoint and rebuilds the model
    if arch == 'vgg':
        model = models.vgg19(pretrained=True)
    elif arch == 'resnet':
        model = models.resnet101(pretrained=True)
        
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print('load checkpoint')
        
    if filepath:
        if os.path.isfile(filepath):
            print('filepath')
            print(filepath)
            checkpoint = torch.load(filepath)
            start_epoch = checkpoint['epochs']
            model.classifier = checkpoint['classifier']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model.class_to_idx = checkpoint['class_to_idx']
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filepath, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))
    return model
        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
    size[0]//2 - 112,
    size[1]//2 - 112,
    size[0]//2 + 112,
    size[1]//2 + 112)
    )
    
    np_image = np.array(image)
    #Scale Image per channel
    # Using (image-min)/(max-min)
    np_image = np_image/255.

    img_a = np_image[:,:,0]
    img_b = np_image[:,:,1]
    img_c = np_image[:,:,2]

    # Normalize image per channel
    img_a = (img_a - 0.485)/(0.229) 
    img_b = (img_b - 0.456)/(0.224)
    img_c = (img_c - 0.406)/(0.225)

    np_image[:,:,0] = img_a
    np_image[:,:,1] = img_b
    np_image[:,:,2] = img_c

    # Transpose image
    np_image = np.transpose(np_image, (2,0,1))
    #print('Finish process image')
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    #print (type(image))
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    #print('show image')
    ax.imshow(image)
    
    return ax

def predict(image_path, model,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    idx_to_class = {}
    for name, label in model.class_to_idx.items():
        idx_to_class[label] = name
    
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.cuda.FloatTensor([image])
    model.eval()
    output = model.forward(Variable(image))
    ps = torch.exp(output.cpu()).data.numpy()[0]

    topk_index = np.argsort(ps)[-topk:][::-1] 
    topk_class = [idx_to_class[x] for x in topk_index]
    topk_prob = ps[topk_index]

    return topk_prob, topk_class

def sanitycheck(classname,prob,category_name):
    # TODO: Display an image along with the top 5 classes
    class_name = []
    cat_to_name = map_labels(category_name)
    for c in classname:
        #print('Class Name ',cat_to_name[c])
        class_name.append(cat_to_name[c])

    class_name.sort(reverse=True)
    return class_name[0]
    