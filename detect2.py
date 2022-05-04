import cv2
import torch
from dialmodel import IRModel
from torchvision import transforms
import os
import numpy as np
import sklearn.preprocessing as sp
from time import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = IRModel().to(device)

weight = r'D:\杨震\unet1\unet\epoch_109_.3f.pt'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))
img_path = 'ROAD3/images/153163_sat.jpg'
mask_path = 'ROAD3/label/153163_mask.png'
mask_path1 = 'ROAD3/label/153163_mask.png'
transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                         std=[0.2479, 0.2475, 0.2485])
])
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
if __name__ == '__main__':
    img_tensor_list = []
    origin = cv2.imread(img_path, 1)
    origin1 = cv2.imread(img_path, 1)


    cv2.imshow('origin', origin)
    tr = transforms.Compose([transforms.ToTensor()])
    img = transforms_test(origin)
    img_tensor_list.append(img)
    img1 = transforms_test(origin1)
    img_tensor_list.append(img1)
    img_tensor_list = torch.stack(img_tensor_list, 0)
    T=cv2.imread(mask_path, 0)
    T = cv2.resize(T, (1024,1024))
    mask = tr(T)
    mask1 = tr(cv2.imread(mask_path1, 0))
    mask=mask.to(device)
    # mask1 = mask1.to(device)
    # pred=mask1

    net.eval()

    with torch.no_grad():
        begin_time = time()
        pred = net(img_tensor_list[0:1].cuda())
        end_time = time()
        time = end_time - begin_time
        print('一共运行时间:', time)

    heatmap = pred .squeeze().cpu()
    single_map = heatmap
    hm = single_map.detach().numpy()


    hm = normalization(hm)
    #
    bin = sp.Binarizer(threshold=0.65)
    hm = bin.transform(hm)

    hm = np.uint8(255 * hm)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (1024,1024))

    superimposed_img = hm
    cv2.imwrite("output/%d.tiff" % 1, superimposed_img)

    pred[pred >= 0.4] = 1
    pred[pred < 0.4] = 0

    TP = ((pred == 1) & (mask == 1)).sum()
    TN = ((pred == 0) & (mask == 0)).sum()
    FN = ((pred == 0) & (mask == 1)).sum()
    FP = ((pred == 1) & (mask == 0)).sum()
    P=TP/(TP+FP)
    pa = (TP + TN) / (TP + TN + FP + FN)
    iou = TP / (TP + FP + FN)
    R=TP/(TP+FN)
    F1=(2*P*R)/(P+R)
    print('pa: ', pa)
    print('R: ', R)
    print('P: ', P)
    print('F1: ', F1)

    print('iou', iou)

    # cv2.imshow('origin_out', np.hstack([img, pred]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

