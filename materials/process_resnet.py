import json
from utils import get_train_classes
import torch
import torch.nn.functional as F

p = torch.load('/home/tyc/zsl-1/materials/model/resnet50-fc-model.bin')
w = p['fc.weight'].data
b = p['fc.bias'].data

v = torch.cat([w, b.unsqueeze(1)], dim=1).tolist()
train_class = get_train_classes()
obj = []
for i in range(len(train_class)):
    obj.append((train_class[i], v[i]))

json.dump(obj, open('/home/tyc/zsl-1/materials/fc-weights.json', 'w'))