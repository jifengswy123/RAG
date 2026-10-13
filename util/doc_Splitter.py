import docx
from typing import List, Tuple

class DocumentSplitter:
    def __init__(self):
        pass

    @staticmethod
    def split_docx(file: str) -> List[Tuple[str, int]]:
        """
        读取docx文档，按段落切分（以整行空行为分界），返回段落内容和段落序号

        参数:
            file_path: docx文件的路径

        返回:
            List[Tuple[str, int]]: 每个元素为(段落原文, 段落序号)的列表
            段落序号从1开始计数
        """
        # 读取文档
        doc = docx.Document(file)

        paragraphs_data = []
        current_paragraph = ""
        paragraph_count = 0

        for para in doc.paragraphs:
            text = para.text.strip()

            if text:  # 非空行
                if current_paragraph:  # 已有内容，添加换行
                    current_paragraph += "\n" + text
                else:  # 新段落开始
                    current_paragraph = text
            else:  # 空行（段落分隔符）
                if current_paragraph:  # 当前段落结束
                    paragraph_count += 1
                    paragraphs_data.append((current_paragraph, paragraph_count))
                    current_paragraph = ""  # 重置当前段落

        # 处理文档末尾的段落（如果没有以空行结尾）
        if current_paragraph:
            paragraph_count += 1
            paragraphs_data.append((current_paragraph, paragraph_count))

        return paragraphs_data

    @staticmethod
    def split_txt(file: str) -> List[Tuple[str, int]]:
        pass

    @staticmethod
    def split_pdf(file: str) -> List[Tuple[str, int]]:
        pass


# 使用方法示例
if __name__ == "__main__":
    # 替换为你的docx文件路径
    file_path = r"../Doc/Win10下docker安装.docx"

    try:
        splitter = DocumentSplitter()
        result = splitter.split_docx(file_path)

        # 打印结果
        for content, index in result:
            print(f"段落 {index}:")
            print(content)
            print("-" * 50)

        print(f"\n总共找到 {len(result)} 个段落")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
    except Exception as e:
        print(f"错误: {e}")