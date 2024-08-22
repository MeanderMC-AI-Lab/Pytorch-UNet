import argparse
import glob
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

# def get_output_filenames(args):
#     def _generate_name(fn):
#         return f'{os.path.splitext(fn)[0]}_OUT.png'

#     return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def stats(image_prediction, image_GT):
    TP = np.sum((image_prediction == True) & (image_GT == 1))  # True Positives
    FP = np.sum((image_prediction == True) & (image_GT == 0))  # False Positives
    FN = np.sum((image_prediction == False) & (image_GT == 1))  # False Positives
    TN = np.sum((image_prediction == False) & (image_GT == 0))  # True Positives
    # # Accuracy 
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)  * 100    
    # # Precision
    # Precision = 100*TP / (TP + FP) if (TP + FP) > 0 else 0 
    #     # Recall 
    # Recall = TP / (TP + FN) * 100
    # Dice
    Dice = 2*TP / (2*TP + FP + FN) 
    # Area
    # Area = 100*np.sum(image_prediction == 1) / np.sum(image_GT == 1)
    return Dice  #Accuracy, Precision, Recall, Dice , Area

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/data/TWO_23_019/TWO_23_019_tmp/Manual_labelling/tmp_data/tmp_UNET_model/w27g3dz2best_trainedUNet.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images or directory', required=True)
    parser.add_argument('--output_dir', '-o', metavar='OUTPUT', default = '/data/TWO_23_019/TWO_23_019_tmp/Manual_labelling/tmp_data/tmp_UNET_predictions' , help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--dir_mask_GT', type=str, default='/data/TWO_23_019/TWO_23_019_tmp/Manual_labelling/UNET_fundo_only_dataset/labels/test', help='Path to dir of GT masks, only required when trying to eval Dice for blood detection')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    if os.path.isdir(in_files[0]): # if it is a directory, find all images in the directory 
        in_files = sorted(glob.glob(in_files[0] +'/*.png'))
    
    # # out_files = get_output_filenames(args)   
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    Dice_list = [] 
    filenames_GT_masks = sorted(glob.glob(args.dir_mask_GT +'/*.png'))
    for i, filename in enumerate(in_files):       
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = os.path.join(args.output_dir,filename.split('/')[-1])
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)

        # Eval prediction
        image_prediction = np.array(Image.open(out_filename))
        image_GT = np.array(Image.open(filenames_GT_masks[i]))
        Dice = stats(image_prediction, image_GT)
        Dice_list.append(Dice)
        
        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
    
    print(f'Dice score is: {np.mean(Dice_list)}.')
