from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as T
from dataset import *
from utils import *
from train import *
from test import predict_image_mask_scoreacc, test_score_acc
import argparse
from model import UnetResnet34
import random
import wandb
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
config = load_train_config('./config.yaml')
parser.add_argument('--CONTINUE_TRAIN', default=True, action='store_false', help='boolean, True False')
parser.add_argument('--EPOCH', type=int, default=30)
args = parser.parse_args()

wandb.init(project='resnet34-unet', config=config)
globals().update(config)
wandb.config.update(config)

if __name__ == "__main__":
        
    #split data       
    df = create_df(DATASET["IMAGE_PATH"])
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

    #datasets
    train_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_train, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"], datasetType='TRAIN', transform=True, patch=False)
    val_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_val, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"], datasetType='VAL', transform=True, patch=False)
    test_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_test, datasetType='TEST', transform=True)

    #dataloader
    train_loader = DataLoader(train_set, batch_size=PARAM["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=PARAM["BATCH_SIZE"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #training params
    model = UnetResnet34(num_classes=PARAM["NUM_CLASSES"]).to(device)
    min_val_loss = np.Inf
    init_lr = PARAM['INITIAL_LR']
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=PARAM['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr, epochs=args.EPOCH, steps_per_epoch=len(train_loader))
    epoch = args.EPOCH
    stop_epoch = PARAM["STOP_EPOCH"]
    init_train = args.CONTINUE_TRAIN

    if not init_train:
        model, optimizer, scheduler, min_val_loss = load_checkpoint(f'{DATASET["MODEL_PATH"]}/best.pth', f'{DATASET["MODEL_PATH"]}/checkpoint.pth', model, optimizer, scheduler, best_checkpoint=False)

    trainval_history = fit(device, epoch, stop_epoch, min_val_loss, model, f'{DATASET["MODEL_PATH"]}', train_loader, val_loader, optimizer, scheduler)

    # plot graph
    plot(trainval_history, 'loss')
    plot(trainval_history, 'score')
    plot(trainval_history, 'acc')
    
    # # test
    bestmodel, optimizer, scheduler, min_val_loss = load_checkpoint(f'{DATASET["MODEL_PATH"]}/best.pth', f'{DATASET["MODEL_PATH"]}/checkpoint.pth', model, optimizer, scheduler, best_checkpoint=True)
    test_score, test_iou = test_score_acc(bestmodel, device, test_set, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"])
    print(f'Test score: {test_score}, Test IOU: {test_iou}')

    rand_selected_img = random.sample(range(len(test_set)), 5)
    test_score=[]
    rand_selected_img.sort()
    print(f'random selected image num: {rand_selected_img}')
    
    for i in rand_selected_img:
        image, mask = test_set[i]
        masked, score, acc = predict_image_mask_scoreacc(device, model, image, mask, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"])
        image.save(f"{DATASET['PREDICT_PATH']}image_{i}.png")
        torch.save(mask,f"{DATASET['PREDICT_PATH']}gt_mask_{i}.pt")
        torch.save(masked,f"{DATASET['PREDICT_PATH']}pred_mask_{i}.pt")
        test_score.append(score)

    for i, num in enumerate(rand_selected_img):
        visualize(
            image=Image.open(f'{DATASET["PREDICT_PATH"]}image_{num}.png'),
            mask=torch.load(f'{DATASET["PREDICT_PATH"]}gt_mask_{num}.pt'),
            pred_mask=torch.load(f'{DATASET["PREDICT_PATH"]}pred_mask_{num}.pt'),
            score=test_score[i])
        torch.cuda.empty_cache()
