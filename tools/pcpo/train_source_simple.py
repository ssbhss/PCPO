#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import os.path as osp
import sys
import random
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from network.ResNet import ResNet


class InverseLR:
    """
    Inverse learning rate scheduler: ε = ε0 · (1 + 10 · e/E)^(-0.75)
    """
    def __init__(self, optimizer, total_epochs, power=0.75):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.power = power
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            progress = self.current_epoch / self.total_epochs
            lr = self.base_lrs[i] * (1 + 10 * progress) ** (-self.power)
            param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]

# Import training config
try:
    from configs.source_training_config import get_training_config
except ImportError:
    def get_training_config(dataset_name):
        return {
            'max_iterations': 10000,
            'batch_size': 64,
            'lr': 0.01,
            'print_freq': 500
        }


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(dataset_name):
    """Get data transforms for different datasets"""
    if dataset_name.lower() == 'visda17':
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    return train_transform, test_transform


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
    
    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data
    
    def __len__(self):
        return len(self.data_loader)


def main(args):
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset-specific training config if not specified
    if args.use_auto_config:
        dataset_config = get_training_config(args.dataset)
        if args.epochs == 100:  # Default value
            args.epochs = dataset_config['epochs']
        if args.batch_size == 64:  # Default value
            args.batch_size = dataset_config['batch_size']
        if args.lr_backbone == 1e-3:  # Default value
            args.lr_backbone = dataset_config['lr_backbone']
        if args.lr_classifier == 1e-2:  # Default value
            args.lr_classifier = dataset_config['lr_classifier']
        if args.momentum == 0.9:  # Default value
            args.momentum = dataset_config['momentum']
        if args.weight_decay == 1e-3:  # Default value
            args.weight_decay = dataset_config['weight_decay']
        if args.print_freq == 100:  # Default value
            args.print_freq = dataset_config.get('print_freq', 100)
        
        print(f"Using auto-config for {args.dataset} (论文标准参数):")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Backbone LR: {args.lr_backbone}")
        print(f"  Classifier LR: {args.lr_classifier}")
        print(f"  Momentum: {args.momentum}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  LR scheduler: inverse (ε = ε0 · (1 + 10 · e/E)^(-0.75))")
        print(f"  Print frequency: {args.print_freq}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get transforms
    train_transform, test_transform = get_transforms(args.dataset)

    if args.dataset == 'visda17' and args.source_domain == 'synthetic':
        # 将 args.source_data_dir 中的 synthetic 替换为 train
        args.source_data_dir = args.source_data_dir.replace('synthetic', 'train')
    # Load dataset using ImageFolder
    print(f"Loading source domain: {args.source_domain}")
    print(f"Data directory: {args.source_data_dir}")
    
    if not os.path.exists(args.source_data_dir):
        raise FileNotFoundError(f"Source data directory not found: {args.source_data_dir}")
    
    train_dataset = ImageFolder(
        root=args.source_data_dir,
        transform=train_transform
    )
    test_dataset = ImageFolder(
        root=args.source_data_dir,
        transform=test_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Verify number of classes matches
    if len(train_dataset.classes) != args.num_classes:
        print(f"Warning: Expected {args.num_classes} classes, found {len(train_dataset.classes)}")
        print("Using actual number of classes found in data")
        args.num_classes = len(train_dataset.classes)
    
    # Create model
    if args.dataset.lower() == 'visda17':
        model = ResNet(101, args.num_classes, pretrained=True)
    else:
        model = ResNet(50, args.num_classes, pretrained=True)
    
    model = model.to(device)
    
    # Loss and optimizer with different learning rates for backbone and classifier
    criterion = nn.CrossEntropyLoss()
    
    # Separate parameters for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:  # classifier parameters
            classifier_params.append(param)
        else:  # backbone parameters
            backbone_params.append(param)
    
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': classifier_params, 'lr': args.lr_classifier}
    ], momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Use inverse learning rate scheduler: ε = ε0 · (1 + 10 · e/E)^(-0.75)
    scheduler = InverseLR(optimizer, args.epochs, power=0.75)
    best_acc = 0.0
    best_epoch = 0

    print(f"Training for {args.epochs} epochs")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Total iterations: {args.epochs * len(train_loader)}")
    print(f"LR scheduler: Inverse (ε = ε0 · (1 + 10 · e/E)^(-0.75))")
    print(f"Backbone LR: {args.lr_backbone}")
    print(f"Classifier LR: {args.lr_classifier}")
    print(f"Momentum: {args.momentum}")
    print(f"Weight decay: {args.weight_decay}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress more frequently for small datasets
            if batch_idx % args.print_freq == 0:
                backbone_lr = optimizer.param_groups[0]['lr']
                classifier_lr = optimizer.param_groups[1]['lr']
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, '
                      f'Backbone LR: {backbone_lr:.6f}, Classifier LR: {classifier_lr:.6f}')
        
        # Update learning rate after each epoch
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        backbone_lr = optimizer.param_groups[0]['lr']
        classifier_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Train Acc: {epoch_acc:.2f}%")
        print(f"  Backbone LR: {backbone_lr:.6f}")
        print(f"  Classifier LR: {classifier_lr:.6f}")
        
        # 测试集评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100. * test_correct / test_total
        print(f"  Test Acc: {test_acc:.2f}%")
        # 用测试集acc保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            model_path = osp.join(args.output_dir, f"best_model_{args.dataset}_{args.source_domain}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved (Test Acc): {best_acc:.2f}%")
        # Stop if no improvement for 10 epochs and accuracy >= 99.5%
        if epoch - best_epoch >= 10 and best_acc >= 99.5:
            print(f"  Early stopping: no improvement for 10 epochs and test accuracy {best_acc:.2f}% >= 99.5%")
            break
    
    print(f"\nTraining completed. Best test accuracy: {best_acc:.2f}%")

    # Save final results
    results_file = osp.join(args.output_dir, f"source_training_results_{args.dataset}_{args.source_domain}.txt")
    with open(results_file, "w") as f:
        f.write(f"Source Domain Training Results\n")
        f.write(f"==============================\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Source Domain: {args.source_domain}\n")
        f.write(f"Architecture: {'ResNet101' if args.dataset.lower() == 'visda17' else 'ResNet50'}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Number of Classes: {args.num_classes}\n")
        f.write(f"Dataset Size: {len(train_dataset)} samples\n")
        f.write(f"Batches per Epoch: {len(train_loader)}\n")
        f.write(f"Total Iterations: {args.epochs * len(train_loader)}\n")
        f.write(f"Best Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Classes: {train_dataset.classes}\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train source domain model for PCPO (Simple version)')
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
    
    # Training parameters
    parser.add_argument('--use-auto-config', action='store_true', default=True,
                       help='Use dataset-specific auto configuration')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--print-freq', type=int, default=100,
                       help='Print frequency (batches)')
    
    # Learning rate parameters
    parser.add_argument('--lr-backbone', type=float, default=1e-3,
                       help='Learning rate for backbone (default: 1e-3)')
    parser.add_argument('--lr-classifier', type=float, default=1e-2,
                       help='Learning rate for classifier (default: 1e-2)')
    
    # Optimizer parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (default: 1e-3)')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)