EMBEDDING_CONFIG = {
    'host': 'http://127.0.0.1:11434',
    'model': 'qwen3-embedding:0.6b',
    'timeout': 30
}

MILVUS_CONFIG = {
    'uri': 'http://127.0.0.1:19530',
    'token': 'swy:Shiweiye1.',
    'db_name': 'MyAILearning',
    'collection_name': 'demo',
    'dimension': 1024
}

def get_embedding_config():
    return EMBEDDING_CONFIG.copy()

def get_milvus_config():
    return MILVUS_CONFIG.copy()

target_directory = r"G:\Program Files\知识库"