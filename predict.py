from dataset import *
from utils import *
from train import *
from test import test_score_acc, predict_image_mask_scoreacc
import argparse
import random
from sklearn.model_selection import train_test_split
from model import UnetResnet34
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
config = load_train_config('./config.yaml')
args = parser.parse_args()
globals().update(config)

if __name__ == "__main__":
    # test

    #split data       
    df = create_df(DATASET["IMAGE_PATH"])
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)

    #datasets
    test_set = DroneDataset(DATASET["IMAGE_PATH"], DATASET["MASK_PATH"], X_test, datasetType='TEST', transform=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_set, batch_size=PARAM["BATCH_SIZE"], shuffle=True)

    model = UnetResnet34(num_classes=PARAM["NUM_CLASSES"]).to(device)
    min_val_loss = np.Inf
    init_lr = PARAM['INITIAL_LR']
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=PARAM['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, init_lr, epochs=PARAM['EPOCH'], steps_per_epoch=len(test_loader))
    

    bestmodel, optimizer, scheduler, min_val_loss = load_checkpoint(f'{DATASET["MODEL_PATH"]}/best.pth', f'{DATASET["MODEL_PATH"]}/checkpoint.pth', model, optimizer, scheduler, best_checkpoint=True)
    test_score, test_iou = test_score_acc(bestmodel, device, test_set, mean=PREPROCESS["MEAN"], std=PREPROCESS["STD"])
    print(f'Test score: {test_score}, Test IOU: {test_iou}')

    rand_selected_img = random.sample(range(len(test_set)), 2)
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
        plt.show(block=False)
        visualize(
            image=Image.open(f'{DATASET["PREDICT_PATH"]}image_{num}.png'),
            mask=torch.load(f'{DATASET["PREDICT_PATH"]}gt_mask_{num}.pt'),
            pred_mask=torch.load(f'{DATASET["PREDICT_PATH"]}pred_mask_{num}.pt'),
            score=test_score[i])