from connectLLM import ask_model
from searchMilvus import search_similar_documents

if __name__ == "__main__":
    user_question = input("请输入你的问题：")
    milvus_docs = search_similar_documents(user_question,2)
    system_prompt=("你是一个热心且专业的系统管理员，擅长各种操作系统的理论知识和中间键安装步骤。"
                   "不要使用milvus_docs之外的知识。"
                   "请判断用户问题user_question和参考文档milvus_docs的内容是否相关。如果只有部分相关，请排除不相关的部分，仅使用相关的部分进行回答"
                   "如果你觉得参考文档与用户问题完全不相关，请回答“我不知道”。"
                   "请先用专业的语气一步一步回答用户提问，回答完成后请列出你使用的参考文档milvus_docs。")
    answer = ask_model(
        user_question=user_question,
        milvus_docs=milvus_docs,
        system_prompt=system_prompt,
        temperature=0.5,
        max_tokens=1000
    )

    print("模型回答：", answer)