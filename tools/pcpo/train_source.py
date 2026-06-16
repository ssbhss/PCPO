#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
import sys
import random
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.data import DataManager
from dassl.engine import build_trainer

from network.ResNet import ResNet


def set_random_seed_custom(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_source_model(args):
    """Train source domain model using standard PyTorch training loop"""
    # Set random seed
    set_random_seed_custom(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup DASSL config for data loading
    cfg = get_cfg_default()
    
    # Map dataset names to DASSL format
    dataset_name_mapping = {
        'office31': 'Office31',
        'office_home': 'OfficeHome',
        'visda17': 'VisDA17',
        'minidomainnet': 'miniDomainNet'
    }
    
    cfg.DATASET.NAME = dataset_name_mapping.get(args.dataset, args.dataset)
    cfg.DATASET.ROOT = osp.dirname(args.source_data_dir)
    cfg.DATASET.SOURCE_DOMAINS = [args.source_domain]
    
    # For source training, we need to provide a dummy target domain
    # We'll use the same domain as source, but DASSL will only load source data for training
    all_domains = {
        'office31': ['amazon', 'dslr', 'webcam'],
        'office_home': ['art', 'clipart', 'product', 'real_world'],
        'visda17': ['synthetic', 'real'],
        'minidomainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    }
    
    # Use a different domain as dummy target
    available_domains = all_domains.get(args.dataset, [args.source_domain])
    dummy_target = [d for d in available_domains if d != args.source_domain]
    if dummy_target:
        cfg.DATASET.TARGET_DOMAINS = [dummy_target[0]]
    else:
        cfg.DATASET.TARGET_DOMAINS = [args.source_domain]  # Fallback
    
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.TRANSFORMS = ["random_flip", "random_translation", "normalize"]
    
    # Build data manager
    dm = DataManager(cfg)
    train_loader = dm.train_loader_x
    
    print(f"Train samples: {len(train_loader.dataset)}")
    
    # Create model
    if args.dataset.lower() == 'visda17':
        model = ResNet(101, args.num_classes, pretrained=True)
    else:
        model = ResNet(50, args.num_classes, pretrained=True)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # Update learning rate
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        # Save model (we'll save every epoch since we don't have validation)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            model_path = osp.join(args.output_dir, f"best_model_{args.source_domain}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: {model_path} (Acc: {best_acc:.2f}%)")
    
    print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
    
    # Save final results
    results_file = osp.join(args.output_dir, f"source_training_results_{args.source_domain}.txt")
    with open(results_file, "w") as f:
        f.write(f"Source Domain Training Results\n")
        f.write(f"==============================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Source Domain: {args.source_domain}\n")
        f.write(f"Architecture: {'ResNet101' if args.dataset.lower() == 'visda17' else 'ResNet50'}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Best Training Accuracy: {best_acc:.2f}%\n")
        f.write(f"Model Path: {model_path}\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train source domain model for PCPO')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['office31', 'office_home', 'visda17', 'minidomainnet'],
                       help='Dataset name')
    parser.add_argument('--source-domain', type=str, required=True,
                       help='Source domain name')
    parser.add_argument('--source-data-dir', type=str, required=True,
                       help='Path to source domain data directory')
    parser.add_argument('--num-classes', type=int, required=True,
                       help='Number of classes')
    parser.add_argument('--output-dir', type=str, default='pretrained_models',
                       help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--step-size', type=int, default=20,
                       help='Step size for learning rate scheduler')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    args = parser.parse_args()
    train_source_model(args)