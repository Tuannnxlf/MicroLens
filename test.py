# 0430
import torch

file_path = "/opt/data/private/vllm2rec/MicroLens/Code/VideoRec/SASRec/gate_analysis/gate/ep_1_bz_1.pt"

try:
    # 加载文件内容
    data = torch.load(file_path)
    print("File loaded successfully!")
    
    # 打印数据信息
    print(f"Data type: {type(data)}")
    
    if isinstance(data, torch.Tensor):
        print(f"Tensor shape: {data.shape}")
        print(f"Tensor dtype: {data.dtype}")
        print("Tensor values (first few):", data.flatten()[:10])  # 打印前10个值（如果是大张量）
    else:
        print("Data content:", data)  # 如果是字典、列表等结构
    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error loading file: {e}")
# old
# import torch
# import random
# from scipy.spatial.distance import cosine
# from tqdm import tqdm  # 导入 tqdm

# # 加载embedding tensor
# def load_embeddings(embeddings_path):
#     # 假设embeddings是一个已经加载的tensor
#     # 如果embeddings是存储在文件中的，可以用torch.load加载
#     embeddings = torch.load(embeddings_path)
#     return embeddings

# # 读取序列数据
# def read_sequences(file_path):
#     sequences = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             sequences.append(list(map(int, line.strip().split())))
#     return sequences

# # 计算两两余弦相似度
# def calculate_cosine_similarity(embedding1, embedding2):
#     return 1 - cosine(embedding1, embedding2)

# # 计算序列内部的平均余弦相似度
# def calculate_sequence_similarity(sequence_embedding):
#     sequence_length = sequence_embedding.size(0)
#     similarity_matrix = torch.zeros((sequence_length, sequence_length))
#     for i in range(sequence_length):
#         for j in range(i + 1, sequence_length):
#             similarity = calculate_cosine_similarity(sequence_embedding[i], sequence_embedding[j])
#             similarity_matrix[i, j] = similarity
#             similarity_matrix[j, i] = similarity
#     non_diagonal_elements = similarity_matrix[~torch.eye(similarity_matrix.size(0)).bool()]
#     average_similarity = non_diagonal_elements.mean().item()
#     return average_similarity

# # 随机采样并计算平均余弦相似度
# def random_sampling(embeddings, sample_size):
#     indices = random.sample(range(1, embeddings.size(0)), sample_size)
#     sample_embeddings = embeddings[indices]
#     return sample_embeddings

# def calculate_random_sample_similarity(sample_embeddings):
#     sample_size = sample_embeddings.size(0)
#     sample_similarity_matrix = torch.zeros((sample_size, sample_size))
#     for i in range(sample_size):
#         for j in range(i + 1, sample_size):
#             similarity = calculate_cosine_similarity(sample_embeddings[i], sample_embeddings[j])
#             sample_similarity_matrix[i, j] = similarity
#             sample_similarity_matrix[j, i] = similarity
#     non_diagonal_elements = sample_similarity_matrix[~torch.eye(sample_size).bool()]
#     average_sample_similarity = non_diagonal_elements.mean().item()
#     return average_sample_similarity

# def main():
#     # 文件路径
#     embeddings_path = '/opt/data/private/vllm2rec/data/videotensor.pt'  # 替换为embedding tensor的实际路径
#     sequences_path = '/opt/data/private/vllm2rec/data/sequence_microlens/data.txt'  # 序列数据文件路径

#     # 加载数据
#     embeddings = load_embeddings(embeddings_path)
#     sequences = read_sequences(sequences_path)

#     sum = 0
#     # 提取序列embedding并计算平均余弦相似度
#     sequence_embeddings = [embeddings[sequence] for sequence in sequences]
#     average_similarities = []
#     for seq_embedding in tqdm(sequence_embeddings, desc="计算序列内部的平均余弦相似度"):
#         line_similarities = calculate_sequence_similarity(seq_embedding)
#         sum += line_similarities
#         average_similarities.append(line_similarities)

#     # 随机采样并计算平均余弦相似度
#     sample_size = 100  # 随机采样数量
#     sample_embeddings = random_sampling(embeddings, sample_size)
#     average_sample_similarity = calculate_random_sample_similarity(sample_embeddings)

#     # 输出结果
#     print("序列内部的平均余弦相似度:", sum/len(average_similarities))
#     print("随机采样的平均余弦相似度:", average_sample_similarity)
# if __name__ == "__main__":
#     main()