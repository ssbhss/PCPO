#!/usr/bin/env python3

"""
可视化不同学习率调度策略的效果
"""

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import argparse
import sys
import os.path as osp

# Add parent directory to Python path
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from configs.source_training_config import DATASET_TRAINING_CONFIG

def create_dummy_model():
    """创建一个虚拟模型用于测试学习率调度"""
    return torch.nn.Linear(10, 1)

def get_lr_schedule(scheduler_type, optimizer, total_iterations):
    """获取指定类型的学习率调度器"""
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=total_iterations//4, T_mult=1, eta_min=0.0001
        )
    elif scheduler_type == 'cosine_simple':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_iterations, eta_min=0.0001
        )
    elif scheduler_type == 'polynomial':
        def lr_lambda(iteration):
            return (1 - iteration / total_iterations) ** 0.9
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'exponential':
        gamma = (0.01) ** (1.0 / total_iterations)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:  # multistep
        lr_milestones = [int(0.5 * total_iterations), int(0.75 * total_iterations), int(0.9 * total_iterations)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.3)

def visualize_lr_schedules():
    """可视化所有数据集的学习率调度"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    datasets = ['office31', 'office_home', 'visda17', 'minidomainnet']
    
    for idx, dataset in enumerate(datasets):
        config = DATASET_TRAINING_CONFIG[dataset]
        total_iterations = config['max_iterations']
        initial_lr = config['lr']
        scheduler_type = config['lr_scheduler']
        
        # 创建虚拟模型和优化器
        model = create_dummy_model()
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)
        scheduler = get_lr_schedule(scheduler_type, optimizer, total_iterations)
        
        # 记录学习率变化
        lrs = []
        iterations = list(range(total_iterations))
        
        for i in range(total_iterations):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # 绘制学习率曲线
        axes[idx].plot(iterations, lrs, linewidth=2, label=f'{scheduler_type}')
        axes[idx].set_title(f'{dataset.upper()}\n({total_iterations} iterations, {scheduler_type})')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Learning Rate')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
        
        # 设置y轴为对数刻度以更好地显示变化
        axes[idx].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('lr_schedules_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("学习率调度可视化已保存为 'lr_schedules_comparison.png'")

def compare_all_schedulers():
    """比较所有调度器在相同设置下的效果"""
    total_iterations = 10000
    initial_lr = 0.01
    
    schedulers = ['multistep', 'cosine', 'cosine_simple', 'polynomial', 'exponential']
    
    plt.figure(figsize=(12, 8))
    
    for scheduler_type in schedulers:
        model = create_dummy_model()
        optimizer = optim.SGD(model.parameters(), lr=initial_lr)
        scheduler = get_lr_schedule(scheduler_type, optimizer, total_iterations)
        
        lrs = []
        for i in range(total_iterations):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        plt.plot(range(total_iterations), lrs, linewidth=2, label=scheduler_type)
    
    plt.title('Learning Rate Schedulers Comparison\n(10000 iterations, initial LR=0.01)')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('all_schedulers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("所有调度器比较已保存为 'all_schedulers_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description='可视化学习率调度策略')
    parser.add_argument('--mode', type=str, default='datasets', 
                       choices=['datasets', 'all'],
                       help='可视化模式: datasets(数据集特定) 或 all(所有调度器比较)')
    
    args = parser.parse_args()
    
    if args.mode == 'datasets':
        visualize_lr_schedules()
    else:
        compare_all_schedulers()

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"需要安装matplotlib: pip install matplotlib")
        print(f"错误: {e}")
    except Exception as e:
        print(f"运行出错: {e}")