import torch
from torchvision import transforms, datasets
import torch.nn.functional as F
import random
import model 

def get_model(name, num_classes):
    if name == 'vgg16':
        classifier = model.vgg16(num_classes)
    elif name == 'resnet50':
        classifier = model.resnet50(num_classes)
    else : 
        classifier = None
        print('the model is not yet implemented! try to another model.')
    return classifier

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_path = './data/'
batch_size = 64
train_folder = 'train/'
test_folder = 'test/'
image_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop((128,128)),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop((128,128)),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
}
set_seed(args.seed)

train_images = datasets.ImageFolder(data_path+train_folder,image_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_images, batch_size = 64 ,num_workers=4,shuffle=True)
test_images = datasets.ImageFolder(data_path+test_folder,image_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_images, batch_size = 64 ,num_workers=4,shuffle=True) 

classifier = get_model('vgg16',1000)
classifier = model.cuda()

lr = 0.0001
lossfn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=student.parameters(),
                            lr=lr)
nepoch = 100
best_score = 0

for epoch in range(nepoch):
    for i, data in enumerate(train_loader):
        img, label = data
        img = img.to(device)
        label = label.to(device)

        pred_y = model(img)
        loss = lossfn(pred_y, label)
        loss.backward()
        optim.step()
    print(loss)




