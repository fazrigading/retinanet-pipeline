import torch
import argparse

from tqdm import tqdm
from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_WORKERS, 
    VALID_IMG,
    VALID_ANNOT,
    TEST_IMG,
    TEST_ANNOT,
    RESIZE_TO,
    CLASSES,
    PROJECT_NAME
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

parser = argparse.ArgumentParser()
parser.add_argument(
    '--weights',
    dest='weights',
    default=f'runs/training/{PROJECT_NAME}/best_model.pth',
    help='path to the model weights'
)
parser.add_argument(
    '--split',
    dest='split',
    default='valid',
    help='data split used for eval'
)
parser.add_argument(
    '--input-images',
    dest='input_images',
    default=None, #'data/Test/Test/JPEGImages',
    help='path to the evaluation images'
)
parser.add_argument(
    '--input-annots',
    dest='input_annots',
    default=None, #'data/Test/Test/JPEGImages',
    help='path to the evaluation annotations'
)
parser.add_argument(
    '--batch',
    dest='batch',
    default=8,
    help='batch size for the data loader'
)
args = parser.parse_args()

# Evaluation function
def validate(valid_data_loader, model):
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach()
            true_dict['labels'] = targets[i]['labels'].detach()
            preds_dict['boxes'] = outputs[i]['boxes'].detach()
            preds_dict['scores'] = outputs[i]['scores'].detach()
            preds_dict['labels'] = outputs[i]['labels'].detach()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################
    

    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.weights, weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    if args.input_images and args.input_annots is not None:
        eval_imgs, eval_annots = (args.input_images, args.input_annots)
    elif args.split == 'test':
        eval_imgs, eval_annots = (TEST_IMG, TEST_ANNOT)
    else:
        eval_imgs, eval_annots = (VALID_IMG, VALID_ANNOT)
        

    test_dataset = create_valid_dataset(
        eval_imgs, 
        eval_annots,
        CLASSES,
        RESIZE_TO
    )
    test_loader = create_valid_loader(
        test_dataset, 
        int(args.batch),
        num_workers=NUM_WORKERS,
    )
    metric = MeanAveragePrecision()
    metric.warn_on_many_detections=False
    
    metric_summary = validate(test_loader, model)
    print(f"mAP_50: {metric_summary['map_50']*100:.3f}")
    print(f"mAP_50_95: {metric_summary['map']*100:.3f}")
    class_counter = 0
    for i in range(0, len(CLASSES)-1, 1):
        class_counter += 1
        print(f"AP_50_95 {class_counter:<3} {CLASSES[i+1]:<20}: {stats['map_per_class'][i]:.3f}")
    print(f"Avg: {stats['map']:.3f}")
