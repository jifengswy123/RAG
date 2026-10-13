import ollama
from config.config import get_embedding_config

def get_embedding(prompt):
    config = get_embedding_config()
    host = config['host']
    model = config['model']
    timeout = config['timeout']

    # 配置 ollama 客户端
    client = ollama.Client(
        host=host,  # 替换为你的服务器IP和端口
        timeout = timeout
    )

    # 调用嵌入模型
    response = client.embeddings(
        model=model,
        prompt=prompt
    )

    # print(f"向量维度: {len(response['embedding'])}")
    embedding_value = response['embedding']
    # print(f"完整向量: {embedding_value}")
    return embedding_value