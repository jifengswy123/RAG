from openai import OpenAI

MODEL = 'qwen2.5:1.5b'
BASE_URL = 'http://127.0.0.1:11434/v1/'
API_KEY = 'ollama'

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)


def ask_model(user_question: str, milvus_docs: str, system_prompt: str,
              temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """
    连接大模型生成回答

    Args:
        user_question: 用户问题
        milvus_docs: Milvus返回的文档内容
        system_prompt: 系统提示词
        temperature: 生成温度 (0.0-2.0)
        max_tokens: 最大生成长度

    Returns:
        str: 模型的回答
    """
    try:
        # 构建完整的提示信息
        full_prompt = f"""{system_prompt}

参考文档：
{milvus_docs}

用户问题：
{user_question}

请根据以上信息回答："""

        # 调用模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"参考文档：{milvus_docs}\n\n用户问题：{user_question}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )

        # 返回模型的回答
        return response.choices[0].message.content

    except Exception as e:
        return f"调用模型时出错：{str(e)}"