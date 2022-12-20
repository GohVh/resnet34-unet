from sklearn.model_selection import train_test_split
import torch
from dataset import *
from utils import *
from train import *
from test import *
import os
import argparse
from model import UnetResnet34
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
config = load_train_config(args.config_path)
globals().update(config)

if __name__ == "__main__":

    def create_df():
        name = []
        for dirname, _, filenames in os.walk(DATASET["IMAGE_PATH"]):
            for filename in filenames:
                name.append(filename.split('.')[0])
        
        return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

    df = create_df()

    #split data
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

    print(f'Dataset path: {DATASET["IMAGE_PATH"]}')

    #datasets
    train_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_train, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"], datasetType='TRAIN', transform=True, patch=False)
    val_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_val, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"], datasetType='VAL', transform=True, patch=False)
    test_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_test, datasetType='TEST', transform=True)

    #dataloader
    batch_size= 3
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    model = UnetResnet34(num_classes=MODEL["NUM_CLASSES"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #training params
    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_loader))

    history = fit(device, epoch, model, train_loader, val_loader, criterion, optimizer, sched)

    #save model
    torch.save(model,f'{DATASET["MODEL_PATH"]}/UnetResnet34.pt')

    # plot graph
    plot(history, 'loss')
    plot(history, 'score')
    plot(history, 'acc')

    # test
    # score, acc = test_score_acc(model, test_set)

    # visualize test result
    rand_selected_img = random.sample(range(len(test_set)), 5)
    rand_selected_img.sort()
    for i in rand_selected_img:
        image, mask = test_set[i]
        masked, score, acc = predict_image_mask_scoreacc(device, model, image, mask, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"])
        visualize(image, mask, masked, score)
