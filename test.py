import yaml
import torch
from utils import *
from torchvision import transforms as T
from tqdm.notebook import tqdm
  
def predict_image_mask_scoreacc(device, model, image, mask, mean, std):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        acc = pixel_accuracy(output, mask)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked, score, acc

def test_score_acc(model, test_set):
    score_iou, accuracy = [], []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score, acc = predict_image_mask_scoreacc(model, img, mask)
        score_iou.append(score)
        accuracy.append(acc)
    return score_iou, accuracy