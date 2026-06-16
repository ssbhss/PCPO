#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import glob
import re
from collections import defaultdict
import numpy as np

def parse_result_file(result_file):
    """解析结果文件，提取准确率"""
    if not osp.exists(result_file):
        return None
    
    try:
        with open(result_file, 'r') as f:
            content = f.read()
        
        # 查找最终准确率
        patterns = [
            r'Final Accuracy[:\s]+([0-9.]+)%?',
            r'Test accuracy[:\s]+([0-9.]+)%?',
            r'Accuracy[:\s]+([0-9.]+)%?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                acc = float(match.group(1))
                # 如果值大于1，假设是百分比形式
                if acc > 1:
                    acc = acc
                else:
                    acc = acc * 100
                return acc
        
        return None
    except Exception as e:
        print(f"解析文件出错 {result_file}: {e}")
        return None

def get_experiment_results(dataset, output_root="output/pcpo"):
    """获取指定数据集的所有实验结果"""
    dataset_dir = osp.join(output_root, dataset)
    if not osp.exists(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        return {}
    
    results = {}
    
    # 查找所有实验目录
    exp_dirs = glob.glob(osp.join(dataset_dir, "*2*_seed*"))
    
    for exp_dir in exp_dirs:
        exp_name = osp.basename(exp_dir)
        
        # 提取源域和目标域
        match = re.match(r'(.+)2(.+)_seed\d+', exp_name)
        if not match:
            continue
        
        source, target = match.groups()
        
        # 查找结果文件
        result_files = [
            osp.join(exp_dir, "final_results.txt"),
            osp.join(exp_dir, "log.txt"),
            osp.join(exp_dir, "results.txt")
        ]
        
        acc = None
        for result_file in result_files:
            acc = parse_result_file(result_file)
            if acc is not None:
                break
        
        if acc is not None:
            results[f"{source}2{target}"] = acc
        else:
            print(f"无法解析结果: {exp_name}")
    
    return results

def summarize_dataset(dataset):
    """汇总单个数据集的结果"""
    print(f"\n=== {dataset.upper()} 结果汇总 ===")
    
    results = get_experiment_results(dataset)
    
    if not results:
        print(f"没有找到 {dataset} 的结果")
        return None
    
    # 按任务排序
    sorted_tasks = sorted(results.keys())
    
    print(f"任务数量: {len(sorted_tasks)}")
    print("详细结果:")
    
    accuracies = []
    for task in sorted_tasks:
        acc = results[task]
        accuracies.append(acc)
        print(f"  {task}: {acc:.2f}%")
    
    # 计算统计信息
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    print(f"\n统计信息:")
    print(f"  平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  最小准确率: {min_acc:.2f}%")
    print(f"  最大准确率: {max_acc:.2f}%")
    
    return {
        'dataset': dataset,
        'results': results,
        'mean': mean_acc,
        'std': std_acc,
        'min': min_acc,
        'max': max_acc,
        'count': len(accuracies)
    }

def save_summary_to_file(summaries, output_file):
    """保存汇总结果到文件"""
    with open(output_file, 'w') as f:
        f.write("PCPO 实验结果汇总\n")
        f.write("=" * 50 + "\n\n")
        
        for summary in summaries:
            if summary is None:
                continue
                
            dataset = summary['dataset']
            f.write(f"{dataset.upper()} 结果:\n")
            f.write("-" * 30 + "\n")
            
            # 详细结果
            sorted_tasks = sorted(summary['results'].keys())
            for task in sorted_tasks:
                acc = summary['results'][task]
                f.write(f"  {task}: {acc:.2f}%\n")
            
            # 统计信息
            f.write(f"\n统计信息:\n")
            f.write(f"  任务数量: {summary['count']}\n")
            f.write(f"  平均准确率: {summary['mean']:.2f}% ± {summary['std']:.2f}%\n")
            f.write(f"  最小准确率: {summary['min']:.2f}%\n")
            f.write(f"  最大准确率: {summary['max']:.2f}%\n")
            f.write("\n" + "=" * 50 + "\n\n")
        
        # 总体汇总
        if len(summaries) > 1:
            f.write("总体汇总:\n")
            f.write("-" * 30 + "\n")
            
            all_means = [s['mean'] for s in summaries if s is not None]
            if all_means:
                overall_mean = np.mean(all_means)
                overall_std = np.std(all_means)
                f.write(f"所有数据集平均准确率: {overall_mean:.2f}% ± {overall_std:.2f}%\n")
            
            for summary in summaries:
                if summary is not None:
                    f.write(f"  {summary['dataset']}: {summary['mean']:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description='PCPO实验结果汇总')
    parser.add_argument('--dataset', type=str, 
                       choices=['office31', 'office_home', 'visda17', 'minidomainnet'],
                       help='指定数据集')
    parser.add_argument('--all-datasets', action='store_true',
                       help='汇总所有数据集结果')
    parser.add_argument('--output-dir', type=str, default='output/pcpo',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    if args.all_datasets:
        # 汇总所有数据集
        datasets = ['office31', 'office_home', 'visda17', 'minidomainnet']
        summaries = []
        
        for dataset in datasets:
            summary = summarize_dataset(dataset)
            summaries.append(summary)
        
        # 保存到文件
        output_file = osp.join(args.output_dir, 'all_results_summary.txt')
        save_summary_to_file(summaries, output_file)
        print(f"\n总体结果已保存到: {output_file}")
        
        # 打印总体汇总
        print(f"\n=== 总体汇总 ===")
        valid_summaries = [s for s in summaries if s is not None]
        if valid_summaries:
            all_means = [s['mean'] for s in valid_summaries]
            overall_mean = np.mean(all_means)
            overall_std = np.std(all_means)
            print(f"所有数据集平均准确率: {overall_mean:.2f}% ± {overall_std:.2f}%")
            
            for summary in valid_summaries:
                print(f"  {summary['dataset']}: {summary['mean']:.2f}%")
    
    elif args.dataset:
        # 汇总指定数据集
        summary = summarize_dataset(args.dataset)
        
        if summary:
            # 保存到文件
            output_file = osp.join(args.output_dir, args.dataset, f'{args.dataset}_results_summary.txt')
            os.makedirs(osp.dirname(output_file), exist_ok=True)
            save_summary_to_file([summary], output_file)
            print(f"\n结果已保存到: {output_file}")
    
    else:
        print("请指定 --dataset 或 --all-datasets")
        parser.print_help()

if __name__ == '__main__':
    main()