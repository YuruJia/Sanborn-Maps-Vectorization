import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.dice_score import dice_loss
from utils.plot_img_and_mask import plot_loss_peak
from utils.dice_score import multiclass_dice_coeff, dice_coeff

from utils.dataset import SanbornDataset
from model.networks import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from config import config
from utils.metrics import *
from evaluate import evaluate

dir_map = config["map_path"]
dir_label = config["label_path"]
dir_ckp = config["checkpoint-path"]
val_percent = config["val_percent"]
batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
save_checkpoint = config["save_checkpoint"]
model_type = config["model_type"]
in_chn = config["in_chn"]
out_nclass = config["out_nclass"]
t = config["recurrent_t"]
beta1 = config["beta1"]
beta2 = config["beta2"]

multiclass = True if out_nclass > 1 else False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

img_scale = 0.5
amp = False
wandb_log = True


def train():
    # 1. Create dataset
    dataset = SanbornDataset(map_path=dir_map, label_path=dir_label)
    print(len(dataset))

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if wandb_log:
        experiment = wandb.init(project='U-Net-brick-frame', resume='allow', anonymous='must')
        experiment.config.update(
            dict(model_type=model_type, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                 amp=amp))

    logging.info(f'''Starting training:
            Model_type:      {model_type}
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

    # 4. Set up the model, optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if model_type == 'U_Net':
        unet = U_Net(img_ch=in_chn, output_ch=out_nclass)
    elif model_type == 'R2U_Net':
        unet = R2U_Net(img_ch=in_chn, output_ch=out_nclass, t=t)
    elif model_type == 'AttU_Net':
        unet = AttU_Net(img_ch=in_chn, output_ch=out_nclass)
    elif model_type == 'R2AttU_Net':
        unet = R2AttU_Net(img_ch=in_chn, output_ch=out_nclass, t=t)

    optimizer = optim.RMSprop(unet.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # optimizer = optim.Adam(list(unet.parameters()),learning_rate, [beta1, beta2])

    unet.to(device=device)

    # 5. Begin training
    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for map, label in train_loader:
                assert map.shape[1] == unet.img_ch, \
                    f'Network has been defined with {unet.img_ch} input channels, ' \
                    f'but loaded images have {map.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                map = map.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    # SR: Segmentation Result
                    SR = unet(map)
                    loss = criterion(SR, label) \
                           + dice_loss(F.softmax(SR, dim=1).float(),
                                       F.one_hot(label, unet.output_chn).permute(0, 3, 1, 2).float(),
                                       multiclass=multiclass)
                    # loss = criterion(SR, label)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(map.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # if loss.item() > 0.5 and global_step > 300:
                #     plot_loss_peak(map, global_step)
                if wandb_log:
                    experiment.log({

                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # ===================================== Validation ====================================#

                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        histograms = {}
                        for tag, value in unet.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(unet, val_loader, device)
                        scheduler.step(val_score)

                        # acc = acc / length
                        # SE = SE / length
                        # SP = SP / length
                        # PC = PC / length
                        # F1 = F1 / length
                        # JS = JS / length
                        # DC = DC / length
                        # print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        # acc, SE, SP, PC, F1, JS, DC))
                        # scheduler.step(dice_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        if wandb_log:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(map[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(label[0].float().cpu()),
                                    'pred': wandb.Image(torch.softmax(SR, dim=1).argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
        if save_checkpoint:
            dir_ckp.mkdir(parents=True, exist_ok=True)
            torch.save(unet.state_dict(),
                       str(dir_ckp / '{}_batchsize_{}_epoch{}.pth'.format(model_type, batch_size, epoch + 1)))
            logging.info(f'Checkpoint_{model_type}_bathsize_{batch_size}_ {epoch + 1} saved!')


if __name__ == "__main__":
    start = time.time()
    train()
    print(f'{model_type} training consumes {(time.time() - start) / 60} min.')
