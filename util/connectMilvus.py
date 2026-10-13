from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from config.config import get_milvus_config
from typing import List, Dict, Any
import numpy as np
import logging

# 这个类用于将向量和数据插入Milvus
# 1.连接Milvus客户端
# 2.定义字段

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查插入的Data，需符合collection中的格式，第一个参数data需为列表，且data中的每一个元素都是一个字典，这个字典的键为str类型，值可以是任何类型
# fields_schema让这个方法灵活可配置，可以预先配置好验证规则：
# my_schema = {"name": str, "age": int}
# my_data = [{"name": "Alice", "age": 25}]
# 然后调用result = validate_insert_data(my_data, my_schema)验证my_data是否符合my_schema的规则
def validate_insert_data(data: List[Dict[str, Any]], fields_schema) -> bool:
    if not isinstance(data, list):
        raise ValueError("data 必须是列表类型")
    if len(data) == 0:
        raise ValueError("data 不能为空列表")

    # 创建字段名到字段定义的映射
    field_map = {field.name: field for field in fields_schema}

    # 必须包含的字段（除了自增的 id）
    required_fields = ["title", "doc_id", "content", "chunk_index", "vector"]

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 条数据必须是字典类型")

        # 检查是否有多余字段
        for key in item.keys():
            if key not in field_map and key != "id":  # id 是自增的，可以不存在
                raise ValueError(f"第 {i} 条数据包含未知字段: {key}")

        # 检查必须字段是否存在
        for field_name in required_fields:
            if field_name not in item:
                raise ValueError(f"第 {i} 条数据缺少必须字段: {field_name}")

                # 检查各个字段的数据类型和格式
                # 1. 检查 title (VARCHAR)
                title = item.get("title")
                if not isinstance(title, str):
                    raise ValueError(f"第 {i} 条数据的 title 必须是字符串类型")
                if len(title) > field_map["title"].max_length:
                    raise ValueError(f"第 {i} 条数据的 title 长度超过 {field_map['title'].max_length} 字符限制")

                # 2. 检查 doc_id (VARCHAR)
                doc_id = item.get("doc_id")
                if not isinstance(doc_id, str):
                    raise ValueError(f"第 {i} 条数据的 doc_id 必须是字符串类型")
                if len(doc_id) > field_map["doc_id"].max_length:
                    raise ValueError(f"第 {i} 条数据的 doc_id 长度超过 {field_map['doc_id'].max_length} 字符限制")

                # 3. 检查 content (VARCHAR)
                content = item.get("content")
                if not isinstance(content, str):
                    raise ValueError(f"第 {i} 条数据的 content 必须是字符串类型")
                if len(content) > field_map["content"].max_length:
                    raise ValueError(f"第 {i} 条数据的 content 长度超过 {field_map['content'].max_length} 字符限制")

                # 4. 检查 chunk_index (INT32)
                chunk_index = item.get("chunk_index")
                if not isinstance(chunk_index, int):
                    raise ValueError(f"第 {i} 条数据的 chunk_index 必须是整数类型")
                if chunk_index < 0:
                    raise ValueError(f"第 {i} 条数据的 chunk_index 不能为负数")
                # 检查是否在 INT32 范围内
                if chunk_index > 2147483647 or chunk_index < -2147483648:
                    raise ValueError(f"第 {i} 条数据的 chunk_index 超出 INT32 范围")

                # 5. 检查 vector (FLOAT_VECTOR, dim=1024)
                vector = item.get("vector")
                if not isinstance(vector, (list, np.ndarray)):
                    raise ValueError(f"第 {i} 条数据的 vector 必须是列表或 numpy 数组")

                # 转换为列表进行检查
                if isinstance(vector, np.ndarray):
                    vector_list = vector.tolist()
                else:
                    vector_list = vector

                if len(vector_list) != 1024:
                    raise ValueError(f"第 {i} 条数据的 vector 维度必须是 1024，当前维度: {len(vector_list)}")

                # 检查向量中的每个元素是否是浮点数
                for j, value in enumerate(vector_list):
                    if not isinstance(value, (int, float, np.float32, np.float64)):
                        raise ValueError(f"第 {i} 条数据的 vector 第 {j} 个元素不是数值类型: {type(value)}")

    # 额外检查：同一文档的 chunk_index 是否连续且唯一
    # 按 doc_id 分组检查
    doc_chunks = {}
    for item in data:
        doc_id = item["doc_id"]
        chunk_idx = item["chunk_index"]

        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(chunk_idx)

    for doc_id, indices in doc_chunks.items():
        # 检查是否有重复的 chunk_index
        if len(indices) != len(set(indices)):
            raise ValueError(f"文档 {doc_id} 中存在重复的 chunk_index")

        # 检查 chunk_index 是否连续（可选，根据业务需求）
        # 如果业务允许非连续，可以注释掉这部分
        sorted_indices = sorted(indices)
        if sorted_indices != list(range(min(indices), max(indices) + 1)):
            raise ValueError(f"文档 {doc_id} 的 chunk_index 不连续")

    return True


def insert_milvus(data):
    config = get_milvus_config()
    uri = config['uri']
    token = config['token']
    db_name = config['db_name']
    collection_name = config['collection_name']
    dimension = config['dimension']

    if not data or len(data) == 0:
        print("警告: 输入数据为空")
        return

    try:
        # 初始化客户端
        client = MilvusClient(uri=uri, token=token, db_name=db_name)

        # 定义字段模式
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]

        # 定义集合模式
        schema = CollectionSchema(fields=fields, description="SWY的学习笔记")

        # 检查集合是否存在，如果不存在则创建
        if not client.has_collection(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            print(f"集合 '{collection_name}' 创建成功")

            # 创建索引（推荐配置）
            index_params = client.prepare_index_params()

            index_params.add_index(
                field_name="vector",
                index_type="HNSW",  # HNSW 图索引
                metric_type="L2",  # 距离度量：L2（欧氏距离）
                params={
                    "M": 16,  # 每个节点最大连接数（4-64之间，值越大精度越高但内存占用越大）
                    "efConstruction": 200  # 构建索引时的搜索空间大小（越大精度越高但构建越慢）
                }
            )

            client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
        else:
            # 可选: 验证现有集合的模式是否匹配
            print(f"集合 '{collection_name}' 已存在")

        res = client.insert(
            collection_name=collection_name,
            data=data
        )

        client.flush(collection_name=collection_name)

        print(f"数据插入成功! 本次插入数量: {res['insert_count']}")
        print(f"集合{collection_name}中现有数据条数为: {client.get_collection_stats(collection_name=collection_name)['row_count']}")

        # 加载集合到内存
        try:
            # 尝试获取集合状态
            load_state = client.get_load_state(collection_name)
            state_value = load_state.get("state", "")
            # 将状态值转换为字符串并统一大小写
            state_str = str(state_value).lower()
            print("collection加载状态：", load_state)
            if not "loaded" in state_str:
                print(f"集合 '{collection_name}' 未加载，正在加载...")
                client.load_collection(collection_name)
                print(f"集合 '{collection_name}' 加载成功")
            else:
                print(f"集合 '{collection_name}' 已加载")
        except Exception as e:
            print(f"检查集合加载状态时出错: {e}")
            # 如果获取集合失败，尝试直接加载
            try:
                print(f"正在加载集合 '{collection_name}'...")
                client.load_collection(collection_name)
                print(f"集合 '{collection_name}' 加载成功")
            except Exception as load_error:
                print(f"加载集合失败: {load_error}")
                raise

    except Exception as e:
        print(f"插入数据时发生错误: {e}")
        raise
    finally:
        # 关闭客户端连接
        if 'client' in locals():
            client.close()

def delete_milvus(doc_id: str):
    config = get_milvus_config()
    uri = config['uri']
    token = config['token']
    db_name = config['db_name']
    collection_name = config['collection_name']
    dimension = config['dimension']

    if not doc_id or not isinstance(doc_id, str):
        logger.error(f"无效的doc_id: {doc_id}")
        return False

    try:
        # 初始化客户端
        client = MilvusClient(uri=uri, token=token, db_name=db_name)
        res = client.delete(collection_name=collection_name,
                            filter='doc_id == "{}"'.format(doc_id)
                            )
    except Exception as e:
        print(f"Delete failed: {e}")
    finally:
        # 关闭客户端连接
        if 'client' in locals():
            client.close()
            return None
        return None