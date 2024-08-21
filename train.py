import argparse
import logging
import numpy as np 
import os
# import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

# Define the transformations
class JointTransform:
    def __init__(self, seed=5):
        torch.manual_seed(seed)
        
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.vflip = transforms.RandomVerticalFlip(p=0.5)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        )

    def __call__(self, image, mask):
        # Apply random horizontal flip
        if torch.rand(1) < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Apply random vertical flip
        if torch.rand(1) < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # Apply color jitter only to the image
        image = self.color_jitter(image)
        
        return image, mask

def apply_joint_transform_to_batch(batch_images, batch_masks, transform):
    transformed_images = []
    transformed_masks = []
    for img, mask in zip(batch_images, batch_masks):
        img, mask = transform(img, mask)
        transformed_images.append(img)
        transformed_masks.append(mask)
    return torch.stack(transformed_images), torch.stack(transformed_masks)

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 20,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        seed: int = 5,
        save_model: int = 0
):
    
    # 1. Create dataset
    dataset_train = BasicDataset(dir_img_train, dir_mask_train, img_scale)
    dataset_val = BasicDataset(dir_img_val, dir_mask_val, img_scale)

    n_val = int(len(dataset_val))
    n_train = int(len(dataset_train))
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    train_loader = DataLoader(dataset_train, shuffle=True, **loader_args)
    val_loader = DataLoader(dataset_val, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net Blood detection', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
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

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_dice = 0
    
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']                
                images, true_masks = apply_joint_transform_to_batch(images, true_masks, JointTransform(seed))
                
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'step': global_step,
                                'epoch': epoch,
                                # **histograms
                            })
                            if epoch == 1:
                                img_grid = torchvision.utils.make_grid(images, nrow=4)
                                # Check the shape and dtype
                                img_grid = img_grid.permute(1, 2, 0).cpu().numpy()  # Convert from CxHxW to HxWxC
                                img_grid = (img_grid * 255).astype(np.uint8)
                                # Log images to wandb
                                experiment.log({"epoch": epoch, "batch": [wandb.Image(img_grid, caption=f"Batch of 11 images")]})
                            if epoch == epochs: # if final epoch
                                experiment.log({    
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                }})
                        except:
                            print(f"An error occurred while logging to WandB: {e}")
                            # pass

                        # to save best model from run 
                        if save_model == 1:
                            if val_score > best_dice: # if current Dice score is higher than previous highest dice score, save mdoel
                                best_dice = val_score
                                best_model_path = os.path.join('/data/TWO_23_019/TWO_23_019_tmp/Manual_labelling/tmp_data/tmp_UNET_model',wandb.run.name+'best_trainedUNet.pt')
                                torch.save(model.state_dict(), best_model_path)
            
        # If you want to save it (original from milesial)
        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     state_dict['mask_values'] = dataset_train.mask_values
        #     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=11, help='Batch size')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--data_path', type=str, default = '/data/TWO_23_019/CholecSeg8k/CholecSeg8k_blood', help='global path to the dataset') 
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--save_model', type=int, default=1, help='If set to 1, save model')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--seed', type=int, default=120, help='Seed of model')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 0. Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set data paths
    dir_img_train = Path(os.path.join(args.data_path,'images/train'))
    dir_mask_train = Path(os.path.join(args.data_path,'labels/train'))
    
    dir_img_val = Path(os.path.join(args.data_path,'images/val'))
    dir_mask_val = Path(os.path.join(args.data_path,'labels/val'))
    dir_checkpoint = Path('./checkpoints/')

    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        state_dict.pop('mask_values', None) 
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            seed = args.seed,
            save_model = args.save_model
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            seed = args.seed
        )
