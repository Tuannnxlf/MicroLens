from utils.load_data import read_items, read_behaviors
import argparse
import os
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def get_user_seq_dic(args):
    before_item_id_to_keys, before_item_name_to_id = read_items(args)
    item_num, user_seq_dic = read_behaviors(before_item_id_to_keys, before_item_name_to_id, 'Log_file', args)
    return item_num, user_seq_dic

def sample_pairs(item_num: int, user_seq_dic: Dict[str, List[int]], mode: str) -> List[Tuple[int, int]]:
    pairs = []
    target_pairs = 10000
    
    if mode == 'intra-sequence':
        while len(pairs) < target_pairs:
            user_seq = random.choice(list(user_seq_dic.values()))
            if len(user_seq) < 2:
                continue
            item1, item2 = random.sample(user_seq, 2)
            pairs.append((item1, item2))
            
    elif mode == 'consecutive':
        while len(pairs) < target_pairs:
            user_seq = random.choice(list(user_seq_dic.values()))
            if len(user_seq) < 2:
                continue
            pos = random.randint(0, len(user_seq) - 2)
            item1, item2 = user_seq[pos], user_seq[pos + 1]
            pairs.append((item1, item2))

    elif mode == 'global':
        all_items = list(range(1, item_num + 1))
        while len(pairs) < target_pairs:
            item1, item2 = random.sample(all_items, 2)
            pairs.append((item1, item2))
    else:
        raise ValueError(f"unknown mode")
    
    return pairs[:target_pairs]
    
def similarity_analysis(all_pairs: Dict[str, List[Tuple[int, int]]], embedding_path_list: List[str]):
    # Define labels for each embedding type
    embedding_labels = [
        'Pretrained Embeddings',
        'Finetuned Video Embeddings',
        'Finetuned ID Embeddings',
        'Fused Embeddings'
    ]
    
    mode_colors = {
        'global': 'blue',
        'intra-sequence': 'green',
        'consecutive': 'red'
    }
    
    all_results = {}
    
    for i, embedding_path in enumerate(embedding_path_list):
        plt.figure(figsize=(12, 8))
        
        # Load embeddings
        embeddings = np.load(embedding_path)  # shape: [item_num, 2048]
        
        for mode, pairs in all_pairs.items():
            similarities = []
            for item1, item2 in pairs:
                emb1 = embeddings[item1]  # shape: (2048,)
                emb2 = embeddings[item2]  # shape: (2048,)

                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(cos_sim)
            similarities = np.array(similarities)
            
            # Calculate statistics
            mean_sim = np.mean(similarities)
            median_sim = np.median(similarities)
            std_sim = np.std(similarities)
            
            print(f"\n{embedding_labels[i]} - {mode} 相似度统计:")
            print(f"平均值: {mean_sim:.4f}")
            print(f"中位数: {median_sim:.4f}") 
            print(f"标准差: {std_sim:.4f}")
            print(f"最小值: {np.min(similarities):.4f}")
            print(f"最大值: {np.max(similarities):.4f}")
            
            # Plot distribution
            counts, bins = np.histogram(similarities, bins=50)
            plt.plot(bins[:-1], counts, alpha=0.7, color=mode_colors[mode], label=f'{mode}')
            
            # Store results
            if embedding_labels[i] not in all_results:
                all_results[embedding_labels[i]] = {}
            all_results[embedding_labels[i]][mode] = {
                'similarities': similarities,
                'mean': mean_sim,
                'median': median_sim,
                'std': std_sim
            }
        
        # Configure plot
        plt.title(f'Similarity Distribution Comparison ({embedding_labels[i]})')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = f'./similarity_distribution_{embedding_labels[i].lower().replace(" ", "_")}.png'
        plt.savefig(plot_path)
        print(f"\n{embedding_labels[i]} 相似度分布对比图已保存到: {plot_path}")
        plt.close()
    
    return all_results

def save_pairs(pairs: List[Tuple[int, int]], mode: str):
    filename = f'./sampled_pairs_{mode}.npy'
    np.save(filename, np.array(pairs))
    print(f"Saved {len(pairs)} pairs to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_video_no', type=int, default=1)
    parser.add_argument('--max_video_no', type=int, default=19738)
    parser.add_argument('--root_data_dir', type=str, default='/opt/data/private/vllm2rec/data/')
    parser.add_argument('--dataset', type=str, default='MicroLens')
    parser.add_argument('--behaviors', type=str, default='MicroLens-100k_pairs.tsv')
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--min_seq_len', type=int, default=5)
    parser.add_argument('--power', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='analysis')
    args = parser.parse_args()

    item_num, user_seq_dic = get_user_seq_dic(args)

    mode_list = ['global', 'intra-sequence', 'consecutive']
    embedding_path_list = [
        '/opt/data/private/vllm2rec/MicroLens/Code/VideoRec/SASRec/statistical_analysis/embedding/pretrained_embeddings.npy',
        '/opt/data/private/vllm2rec/MicroLens/Code/VideoRec/SASRec/statistical_analysis/embedding/finetuned_video_embeddings.npy',
        '/opt/data/private/vllm2rec/MicroLens/Code/VideoRec/SASRec/statistical_analysis/embedding/finetuned_id_embeddings.npy',
        '/opt/data/private/vllm2rec/MicroLens/Code/VideoRec/SASRec/statistical_analysis/embedding/fused_embeddings.npy'
    ]

    # First sample and save all pairs
    all_pairs = {}
    for mode in mode_list:
        print(f"\n{'='*50}")
        print(f"Sampling pairs for mode: {mode}")
        print(f"{'='*50}")
        pairs = sample_pairs(item_num, user_seq_dic, mode)
        save_pairs(pairs, mode)
        all_pairs[mode] = pairs
    
    # Then perform analysis with all pairs
    print("\nStarting similarity analysis...")
    all_results = similarity_analysis(all_pairs, embedding_path_list)