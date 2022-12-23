from utils import *
import time
from tqdm.notebook import tqdm
import wandb
import torch.nn as nn

def fit(device, epoch, stop_epoch, min_val_loss, model, model_path, train_loader, val_loader, optimizer, scheduler, patch=False):

    criterion = nn.CrossEntropyLoss().to(device)
    wandb.watch(model, criterion, log='all', log_freq=10)

    train_losses, val_losses = [], []
    val_iou, train_iou, val_acc, train_acc = [], [], [], []
    loss_notdecrease_count = 0
    prev_e_loss = min_val_loss

    # model.to(device)
    fit_time = time.time()

    for e in range(epoch):
        since = time.time()
        t_loss_perb, t_iou_perb, t_acc_perb = 0, 0, 0
        v_loss_perb, v_iou_perb, v_acc_perb = 0, 0, 0
        t_loss_pere, v_loss_pere = 0, 0

        #training loop
        model.train()

        for i, data in enumerate(tqdm(train_loader)):
            
            #training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1,c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)
            
            image, mask = image_tiles.to(device), mask_tiles.to(device)
            #forward
            output = model(image)
            loss = criterion(output, mask)
            #evaluation metrics
            t_iou_perb += mIoU(output, mask)
            t_acc_perb += pixel_accuracy(output, mask)
            #backward
            loss.backward()
            optimizer.step() #update weight          
            optimizer.zero_grad()#reset gradient

            #step the learning rate
            scheduler.step() 
            
            t_loss_perb += loss.item()
            torch.cuda.empty_cache()
            
        with torch.no_grad():

            model.eval()
            
            #validation loop
            
            for i, data in enumerate(tqdm(val_loader)):
                #reshape to 9 patches from single image, delete batch size
                image_tiles, mask_tiles = data

                if patch:
                    bs, n_tiles, c, h, w = image_tiles.size()

                    image_tiles = image_tiles.view(-1,c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)
                
                image, mask = image_tiles.to(device), mask_tiles.to(device)
                output = model(image)
                #evaluation metrics
                v_iou_perb +=  mIoU(output, mask)
                v_acc_perb += pixel_accuracy(output, mask)
                #loss
                loss = criterion(output, mask)                                  
                v_loss_perb += loss.item()
                torch.cuda.empty_cache()
            
        #calculation mean for each batch
        t_loss_pere = t_loss_perb/len(train_loader)
        train_losses.append(t_loss_pere)
        v_loss_pere = v_loss_perb/len(val_loader)
        val_losses.append(v_loss_pere)
            
        checkpoint = {
			'epoch': e + 1,
			'train_loss': t_loss_pere,
			'val_loss': v_loss_pere,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
			}

        save_checkpoint(checkpoint, f'{model_path}/checkpoint.pth')
        print('save model...')

        if min_val_loss > v_loss_pere:
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_val_loss, v_loss_pere))
            min_val_loss = v_loss_pere
            save_checkpoint(checkpoint, f'{model_path}/checkpoint.pth',True, f'{model_path}/best.pth')
        
        else:
            if v_loss_pere > prev_e_loss:
                loss_notdecrease_count+=1
                print(f'Loss Not Decrease for {loss_notdecrease_count} time')

                if loss_notdecrease_count==stop_epoch:
                    print(f'Loss not decrease for {stop_epoch} times, Stop Training')
                    break
            
        #iou
        val_iou.append(v_iou_perb/len(val_loader))
        train_iou.append(t_iou_perb/len(train_loader))
        train_acc.append(t_acc_perb/len(train_loader))
        val_acc.append(v_acc_perb/ len(val_loader))
        print("Epoch:{}/{}..".format(e+1, epoch),
                "Train Loss: {:.3f}..".format(t_loss_pere),
                "Val Loss: {:.3f}..".format(v_loss_pere),
                "Train Score:{:.3f}..".format(t_iou_perb/len(train_loader)),
                "Val Score: {:.3f}..".format(v_iou_perb/len(val_loader)),
                "Train Acc:{:.3f}..".format(t_acc_perb/len(train_loader)),
                "Val Acc:{:.3f}..".format(v_acc_perb/len(val_loader)),
                "Time: {:.2f}m".format((time.time()-since)/60))
        
    history = {'train_loss' : train_losses, 'val_loss': val_losses,
               'train_score' :train_iou, 'val_score':val_iou,
               'train_acc' :train_acc, 'val_acc':val_acc}
               
    print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
    return history