from pymilvus import MilvusClient
from config.config import get_milvus_config
from util.useEmbeddingModel import get_embedding


def search_similar_documents(question,top_k):
    #1. 获取Milvus配置参数
    try:
        config = get_milvus_config()
        uri = config['uri']
        token = config['token']
        db_name = config['db_name']
        collection_name = config['collection_name']
        dimension = config['dimension']

        client = MilvusClient(
            uri=uri,  # Milvus实例的地址。
            token=token,  # 登录Milvus实例的用户名和密码。
            db_name=db_name  # 待连接的数据库名称，本文示例为默认的default。
        )

        client.load_collection(
            collection_name=collection_name
        )

        res = client.get_load_state(
            collection_name=collection_name
        )

        print("res的值为:")
        print(res)

        query = question
        query_vector = get_embedding(query)
        print(f"top_k 参数: {top_k}")

        # 设置搜索参数
        search_params = {
            "metric_type": "L2",    # 使用欧氏距离
            "offset": 0,
            "params": {
                "ef": 20,
            }
        }

        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            search_params=search_params,
            limit=top_k,    # limit要设置在search方法中，在search_params中设置无效
            output_fields=["id", "title","content","chunk_index"]
        )
        print(f"搜索结果数量: {len(results[0]) if results and results[0] else 0}")

        # 处理多个搜索结果
        if results and len(results[0]) > 0:
            search_results = []

            for match in results[0]:
                result_item = {
                    "id": match['entity']['id'],
                    "title": match['entity']['title'],
                    "content": match['entity']['content'],
                    "chunk_index": match['entity']['chunk_index'],
                    "similarity": 1 - match['distance'] if 'distance' in match else None,
                    "distance": match.get('distance'),
                    "rank": len(search_results) + 1  # 添加排名
                }
                search_results.append(result_item)

            return {
                "success": True,
                "data": search_results,
                "message": "成功找到相关文档"
            }
        else:
            return {
                "success": False,
                "data": None,
                "error": "未找到相关文档"
            }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"搜索过程中发生错误: {str(e)}"
        }


if __name__ == "__main__":
    question = input("请输入你的问题：")
    result = search_similar_documents(question, 3)

    print("\n搜索结果:")
    if result.get('success'):
        for i, doc in enumerate(result.get('data', []), 1):
            print(f"{i}. [{doc.get('similarity'):.4f}] {doc.get('title')} ")
            print(f"   内容: {doc.get('content', '')[:80]}...")
    else:
        print(f"搜索失败: {result.get('error')}")
