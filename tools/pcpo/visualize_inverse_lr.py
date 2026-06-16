#!/usr/bin/env python3

"""
可视化Inverse学习率调度器
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path as osp

# Add parent directory to Python path
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from configs.source_training_config import get_training_config

def inverse_lr_schedule(lr0, epoch, total_epochs, power=0.75):
    """
    计算inverse学习率: ε = ε0 · (1 + 10 · e/E)^(-0.75)
    """
    progress = epoch / total_epochs
    return lr0 * (1 + 10 * progress) ** (-power)

def visualize_lr_schedules():
    """可视化不同数据集的学习率调度"""
    datasets = ['office31', 'office_home', 'visda17', 'minidomainnet']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        config = get_training_config(dataset)
        epochs = config['epochs']
        lr_backbone = config['lr_backbone']
        lr_classifier = config['lr_classifier']
        
        # 计算学习率曲线
        epoch_range = np.arange(0, epochs)
        backbone_lrs = [inverse_lr_schedule(lr_backbone, e, epochs) for e in epoch_range]
        classifier_lrs = [inverse_lr_schedule(lr_classifier, e, epochs) for e in epoch_range]
        
        # 绘制曲线
        axes[i].plot(epoch_range, backbone_lrs, 'b-', label=f'Backbone (初始: {lr_backbone})', linewidth=2)
        axes[i].plot(epoch_range, classifier_lrs, 'r-', label=f'Classifier (初始: {lr_classifier})', linewidth=2)
        
        axes[i].set_title(f'{dataset.upper()} ({epochs} epochs)', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Learning Rate')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')  # 使用对数刻度更好地显示学习率变化
        
        # 添加一些关键点的标注
        mid_epoch = epochs // 2
        final_epoch = epochs - 1
        
        mid_backbone_lr = inverse_lr_schedule(lr_backbone, mid_epoch, epochs)
        final_backbone_lr = inverse_lr_schedule(lr_backbone, final_epoch, epochs)
        
        axes[i].annotate(f'Mid: {mid_backbone_lr:.2e}', 
                        xy=(mid_epoch, mid_backbone_lr), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        axes[i].annotate(f'Final: {final_backbone_lr:.2e}', 
                        xy=(final_epoch, final_backbone_lr), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('Inverse Learning Rate Schedule: ε = ε0 · (1 + 10 · e/E)^(-0.75)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 保存图片
    output_path = 'inverse_lr_schedule.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"学习率调度可视化已保存到: {output_path}")
    
    # 显示图片
    plt.show()

def compare_lr_schedules():
    """比较不同学习率调度器"""
    epochs = 100
    lr0 = 0.01
    
    epoch_range = np.arange(0, epochs)
    
    # Inverse调度器
    inverse_lrs = [inverse_lr_schedule(lr0, e, epochs) for e in epoch_range]
    
    # Step调度器 (每20个epoch衰减0.1倍)
    step_lrs = []
    for e in epoch_range:
        if e < 20:
            step_lrs.append(lr0)
        elif e < 40:
            step_lrs.append(lr0 * 0.1)
        elif e < 60:
            step_lrs.append(lr0 * 0.01)
        elif e < 80:
            step_lrs.append(lr0 * 0.001)
        else:
            step_lrs.append(lr0 * 0.0001)
    
    # 余弦调度器
    cosine_lrs = [lr0 * 0.5 * (1 + np.cos(np.pi * e / epochs)) for e in epoch_range]
    
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_range, inverse_lrs, 'b-', label='Inverse (论文使用)', linewidth=2)
    plt.plot(epoch_range, step_lrs, 'r-', label='Step (每20epoch衰减)', linewidth=2)
    plt.plot(epoch_range, cosine_lrs, 'g-', label='Cosine', linewidth=2)
    
    plt.title('不同学习率调度器对比', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 保存图片
    output_path = 'lr_scheduler_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"学习率调度器对比已保存到: {output_path}")
    
    plt.show()

def print_lr_analysis():
    """打印学习率分析"""
    print("Inverse学习率调度器分析")
    print("=" * 50)
    print("公式: ε = ε0 · (1 + 10 · e/E)^(-0.75)")
    print()
    
    datasets = ['office31', 'office_home', 'visda17', 'minidomainnet']
    
    for dataset in datasets:
        config = get_training_config(dataset)
        epochs = config['epochs']
        lr_backbone = config['lr_backbone']
        lr_classifier = config['lr_classifier']
        
        print(f"{dataset.upper()} ({epochs} epochs):")
        
        # 计算关键时刻的学习率
        key_epochs = [0, epochs//4, epochs//2, 3*epochs//4, epochs-1]
        
        print("  Backbone学习率变化:")
        for e in key_epochs:
            lr = inverse_lr_schedule(lr_backbone, e, epochs)
            progress = e / epochs * 100
            print(f"    Epoch {e:2d} ({progress:5.1f}%): {lr:.6f}")
        
        print("  Classifier学习率变化:")
        for e in key_epochs:
            lr = inverse_lr_schedule(lr_classifier, e, epochs)
            progress = e / epochs * 100
            print(f"    Epoch {e:2d} ({progress:5.1f}%): {lr:.6f}")
        
        # 计算总的学习率衰减比例
        initial_backbone = lr_backbone
        final_backbone = inverse_lr_schedule(lr_backbone, epochs-1, epochs)
        decay_ratio = final_backbone / initial_backbone
        
        print(f"  Backbone学习率衰减比例: {decay_ratio:.4f} ({decay_ratio*100:.2f}%)")
        print()

if __name__ == "__main__":
    print_lr_analysis()
    print("\n正在生成可视化图表...")
    
    try:
        import matplotlib.pyplot as plt
        visualize_lr_schedules()
        compare_lr_schedules()
    except ImportError:
        print("警告: matplotlib未安装，跳过可视化")
        print("可以运行: pip install matplotlib 来安装")