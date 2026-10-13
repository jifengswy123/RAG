import requests
import json


def rerank_with_proper_format(query: str, documents: list):
    """
    使用正确的格式调用 BGE-reranker
    """
    # BGE-reranker 的推荐格式
    # 格式: query: <query> document: <document>

    formatted_prompts = []
    for doc in documents:
        formatted_prompts.append(f"query: {query} document: {doc}")

    # 构建请求
    data = {
        "model": "qllama/bge-reranker-v2-m3:latest",
        "prompt": "\n".join(formatted_prompts),  # 每个 query-document 对一行
        "stream": False
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=data
    )

    if response.status_code == 200:
        return response.json()
    else:
        return response.text


# 测试
query = "What is machine learning?"
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks for feature learning.",
    "Machine learning can be supervised or unsupervised."
]

result = rerank_with_proper_format(query, documents)
print("结果:", json.dumps(result, indent=2, ensure_ascii=False))