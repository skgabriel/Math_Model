#Import statements 


import pickle, json, cv2, os, sys
import tqdm, torch, torchvision

import numpy as np
from pprint import pprint
from PIL import Image
from torch.utils.data import Dataset 
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_root = './Data/'


class MathData(Dataset):
    """
        Data Loading
    """
    def __init__(self, root, split='Train', transform = transforms.Compose([transforms.ToTensor()]), feature_index=None, cache_json=True):
        self.root = root 
        self.split = split 
        self.transform = transform 
        self.feature_index = feature_index 
        self.cache_json = cache_json
        files = os.listdir(os.path.join(root, split))
        self.name_indices = sorted(
                [f.split('.')[0] for f in files if f[-4:] == '.jpg'])
        if(cache_json):
            self.feature_cache = {}
            iterate = tqdm.tqdm(self.name_indices)
            iterate.set_description('Caching Labels')
            for name_index in iterate:
                self.feature_cache[name_index] = self.read_json_feature(
                        name_index)
    
    def get_file_path(self, suffix, name_index):
        file_name = ('%s.' + '%s')%(name_index, suffix)
        return os.path.join(self.root, self.split, file_name)
    
    def read_json_feature(self, split, name_index):
        file_path = self.get_file_path('json', split + '_labels')
        data = json.load(open(file_path))
        file_name = name_index + '.jpg'
        return [entry for (index, entry) in enumerate(data)
                if entry['image'] == file_name][0]['label']
    
    def __getitem__(self, index):
        name_index = self.name_indices[index]
        dataset = (name_index.split('_'))[0]
        image = Image.open(
                self.get_file_path('jpg', name_index)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        if self.cache_json:
            feature = self.feature_cache[name_index]
        else:
            feature = self.read_json_feature(dataset, name_index)
        
        return image, feature
    
    def __len__(self):
        return len(self.name_indices)

def unity_collate(batch):
    images, features = [], []
    for i, b in enumerate(batch):
        image, feature = b
        images.append(image)
        features.append([encode(feature)])
    
    return torch.stack(images), torch.LongTensor(features)


class Net(nn.Module):
    """
      neural network architecture code 
    """
    def __init__(self, feature_id):
        super(Net, self).__init__()
        self.convs = []
        self.BNs = []
        # Filter sizes (Input/Output)
        io = [(3, 8), (8, 16), (16, 32), (32, 64), (64, 32)]
        for i, o in io:
          self.convs.append(nn.Conv2d(i, o, 5, stride=2, padding=(0,0)).cuda())
          self.BNs.append(nn.BatchNorm2d(o).cuda())
        self.fc1 = nn.Linear(32 * 6 * 6, 128).cuda()
        self.fc2 = nn.Linear(128, 32).cuda()
        self.fc3 = nn.Linear(32, 5) .cuda()
        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()
        
        self.feature_id = feature_id
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4,
                                    weight_decay=1e-4)
    
        self.loss = nn.CrossEntropyLoss()

    def forward(self, images):
        x = images
        for conv, bn in zip(self.convs, self.BNs):
          x = bn(F.relu(conv(x)))
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def train_batch(self, images, labels):
        self.optimizer.zero_grad()
        output = self(images)
        single_feature_labels = labels[:,self.feature_id]
        loss = self.loss(output, single_feature_labels)
        loss.backward()
        
        self.optimizer.step()
        val, preds = torch.max(output, 1)
        return loss.data[0], preds
    
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)


# Data Loading
batch_size = 64
feature_id = 0
train_set = DataLoader(
        UnityImgData(data_root, cache_json=False),
        batch_size = batch_size,
        num_workers = 4,
        shuffle = True,
        collate_fn = unity_collate)
test_set = DataLoader(
        UnityImgData(data_root, cache_json=False, split='test'),
        batch_size = batch_size,
        num_workers = 4,
        shuffle = True,
        collate_fn = unity_collate)

model = Net(feature_id)

def run_eval():
    """
      Evaluation
    """
    model.train(mode=False)
    v_pred = []
    for images, features in test_set:
        image_var = Variable(images).cuda()
        label_var = Variable(features).cuda()
        val, preds = torch.max(model(image_var), 1)
        v_pred.extend([1 if p == g else 0 for p,g in 
                      zip(preds.cpu().data.numpy(), 
                          np.squeeze(features.cpu().numpy()))])
    return sum(v_pred)/len(v_pred)

def train(start_epoch, end_epoch):
    """
      Training, includes a call to evaluation
    """
    model.train()
    for epoch in range(start_epoch, end_epoch):
        model.train(mode=True)
        all_losses = []
        t_pred = []
        iterate = tqdm.tqdm(train_set)
        for images, features in iterate:
            image_var = Variable(images).cuda()
            label_var = Variable(features).cuda()
            loss, preds = model.train_batch(image_var, label_var)
            all_losses.append(loss)
            t_pred.extend([1 if p == g else 0 for p,g in 
                          zip(preds.cpu().data.numpy(), 
                              np.squeeze(features.cpu().numpy()))])
        
        checkpoint_path = 'single_feature_%i.%i.checkpoint'%(feature_id, epoch)
        model.save_model(checkpoint_path)

        print('{}  Train Loss: {:.5f}   Acc: {:.5f}    Test Acc: {:.5f}'
              .format(epoch, sum(all_losses)/len(all_losses),
                      sum(t_pred)/len(t_pred), run_eval()))
    return all_losses 

# If a model is passed in, run evaluation and exit
if len(sys.argv) == 2:
  model = pickle.load(open(sys.argv[1], 'rb'))
  print(run_eval())
  sys.exit()

# Else, train and save
train(0,50)
pickle.dump(model, open('model.pkl','wb'))
