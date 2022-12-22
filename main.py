from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from dataset import *
from utils import *
from train import *
from test import predict_image_mask_scoreacc, test_score_acc
import os
import argparse
from model import UnetResnet34
import random
import wandb
import pandas as pd
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
config = load_train_config(args.config_path)
wandb.init(project=args.project_name, config=config)
globals().update(config)
wandb.config.update(config)

if __name__ == "__main__":
        
    #split data
    def create_df():
        name = []
        for dirname, _, filenames in os.walk(DATASET["IMAGE_PATH"]):
            for filename in filenames:
                name.append(filename.split('.')[0])        
        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

    df = create_df()
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
    model = UnetResnet34(num_classes=MODEL["NUM_CLASSES"])
    min_val_loss = np.Inf
    lr = PARAM['INITIAL_LR']
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=PARAM['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=PARAM['EPOCH'],
                                            steps_per_epoch=len(train_loader))
    epoch = PARAM["EPOCH"]
    stop_epoch = PARAM["STOP EPOCH"]
    init_train = PARAM["INIT_TRAIN"]

    if not init_train:
        model, optimizer, scheduler, min_val_loss = load_checkpoint(f'{DATASET["MODEL_PATH"]}/best.pth', f'{DATASET["MODEL_PATH"]}/checkpoint.pth', model, optimizer, scheduler, best_checkpoint=False)

    trainval_history = fit(device, epoch, stop_epoch, min_val_loss, model, train_loader, val_loader, criterion, optimizer, scheduler)

    # plot graph
    plot(trainval_history, 'loss')
    plot(trainval_history, 'score')
    plot(trainval_history, 'acc')
    
    # test
    bestmodel, optimizer, scheduler, min_val_loss = load_checkpoint(f'{DATASET["MODEL_PATH"]}/best.pth', f'{DATASET["MODEL_PATH"]}/checkpoint.pth', model, optimizer, scheduler, best_checkpoint=True)
    test_history = test_score_acc(bestmodel, test_set)
    plot(test_history, 'score', isTest=True)
    plot(test_history, 'acc', isTest=True)

    # visualize test result
    rand_selected_img = random.sample(range(len(test_set)), 5)
    rand_selected_img.sort()
    for i in rand_selected_img:
        image, mask = test_set[i]
        masked, score, acc = predict_image_mask_scoreacc(device, model, image, mask, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"])
        visualize(image, mask, masked, score)
