#!/usr/bin/env python3
"""
运行引导型话题检测流程的脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from main_pipeline import GuidedTopicDetectionPipeline

def setup_logging():
    """设置日志"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/run_pipeline.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='引导型话题检测系统运行脚本')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], 
                       default='predict', help='运行模式：train（训练）或predict（预测）')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='数据路径')
    parser.add_argument('--config_path', type=str, 
                       default='configs/pipeline_config.json',
                       help='配置文件路径')
    
    # 训练模式参数
    parser.add_argument('--labels_path', type=str, 
                       help='标签文件路径（训练模式必需）')
    
    # 预测模式参数
    parser.add_argument('--model_path', type=str, 
                       help='模型路径（预测模式）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                       default='results',
                       help='输出目录')
    parser.add_argument('--save_intermediate', action='store_true',
                       help='保存中间结果')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 检查参数
    if args.mode == 'train' and not args.labels_path:
        logger.error("训练模式需要提供标签文件路径 --labels_path")
        return
    
    if args.mode == 'predict' and not args.model_path:
        logger.warning("预测模式未提供模型路径，将使用未训练的模型")
    
    # 检查数据路径
    if not os.path.exists(args.data_path):
        logger.error(f"数据路径不存在: {args.data_path}")
        return
    
    # 检查配置文件
    if not os.path.exists(args.config_path):
        logger.error(f"配置文件不存在: {args.config_path}")
        return
    
    try:
        # 创建检测流程
        logger.info("初始化引导型话题检测流程...")
        pipeline = GuidedTopicDetectionPipeline(args.config_path)
        
        if args.mode == 'train':
            # 训练模式
            logger.info("开始训练模式...")
            
            # 加载标签
            labels = []
            with open(args.labels_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        labels.append(int(line))
            
            logger.info(f"加载了 {len(labels)} 个标签")
            
            # 运行训练流程
            result = pipeline.run_full_pipeline(
                args.data_path, 
                labels=labels, 
                save_results=True
            )
            
            if result['success']:
                logger.info("训练完成！")
                metrics = result['results']['performance_metrics']
                print("\n=== 训练结果 ===")
                print(f"测试准确率: {metrics.get('accuracy', 0):.4f}")
                print(f"测试精确率: {metrics.get('precision', 0):.4f}")
                print(f"测试召回率: {metrics.get('recall', 0):.4f}")
                print(f"测试F1分数: {metrics.get('f1_score', 0):.4f}")
                print(f"测试AUC: {metrics.get('auc', 0):.4f}")
                print(f"执行时间: {result['execution_time']:.2f} 秒")
            else:
                logger.error(f"训练失败: {result['error']}")
                return 1
        
        elif args.mode == 'predict':
            # 预测模式
            logger.info("开始预测模式...")
            
            # 加载训练好的模型
            if args.model_path and os.path.exists(args.model_path):
                pipeline.load_trained_model(args.model_path)
                logger.info("已加载训练好的模型")
            
            # 运行预测流程
            result = pipeline.run_full_pipeline(
                args.data_path, 
                save_results=True
            )
            
            if result['success']:
                logger.info("预测完成！")
                predictions = result['predictions']
                probabilities = result['probabilities']
                
                print("\n=== 预测结果 ===")
                print(f"预测标签: {predictions}")
                print(f"预测概率: {probabilities}")
                
                # 统计结果
                guided_count = sum(1 for p in predictions if p == 0)
                normal_count = sum(1 for p in predictions if p == 1)
                
                print(f"\n引导型话题数量: {guided_count}")
                print(f"普通话题数量: {normal_count}")
                print(f"执行时间: {result['execution_time']:.2f} 秒")
                
                # 保存预测结果到文件
                output_file = os.path.join(args.output_dir, 'predictions.txt')
                os.makedirs(args.output_dir, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    f.write("预测结果\n")
                    f.write("=" * 50 + "\n")
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        topic_type = "引导型" if pred == 0 else "普通"
                        f.write(f"时间片 {i+1}: {topic_type}话题 (概率: {prob:.4f})\n")
                
                logger.info(f"预测结果已保存到: {output_file}")
            else:
                logger.error(f"预测失败: {result['error']}")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"运行过程中发生错误: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
