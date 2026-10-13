import os
import docx
from typing import List, Dict
import logging
from config.config import target_directory
from util.util import generate_doc_id_by_hash

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentSplitter:
    #self:init方法本身；doc_dir:init方法的参数；str:doc_dir这个参数的数据类型；"Doc":如果不提供参数，默认参数为Doc即文档目录
    def __init__(self, doc_dir: str = "Doc"):
        self.doc_dir = doc_dir
        # 文件类型处理器映射
        self.file_handlers = {
            '.docx': self._process_docx,
            '.doc': self._process_doc,
            # 可以在这里添加其他文件类型的处理器
            # '.pdf': self._process_pdf,
            # '.txt': self._process_txt,
        }

    def split_all_documents(self) -> List[Dict]:
        all_segments = []

        if not os.path.exists(self.doc_dir):
            logger.error(f"目录不存在: {self.doc_dir}")
            return all_segments

        # 遍历目录下的所有文件
        for filename in os.listdir(self.doc_dir):
            file_path = os.path.join(self.doc_dir, filename)

            # 只处理文件，忽略目录
            if not os.path.isfile(file_path):
                continue

            # 获取文件扩展名
            file_ext = os.path.splitext(filename)[1].lower()

            # 检查是否有对应的处理器
            if file_ext in self.file_handlers:
                logger.info(f"处理文件: {filename}")
                try:
                    segments = self.file_handlers[file_ext](file_path, filename)
                    all_segments.extend(segments)
                except Exception as e:
                    logger.error(f"处理文件 {filename} 时出错: {str(e)}")
            else:
                logger.warning(f"不支持的文件类型: {filename}")

        logger.info(f"总共切分出 {len(all_segments)} 个段落")
        return all_segments

    def _process_docx(self, file_path: str, filename: str) -> List[Dict]:
        segments = []

        try:
            # 读取docx文件
            doc = docx.Document(file_path)

            # 生成文档唯一ID（使用文件名）
            doc_id = generate_doc_id_by_hash(filename)

            # 用于存储当前段落块的内容
            current_block = []
            chunk_index = 1  # 段落块在文档中的序号

            for para in doc.paragraphs:
                text = para.text

                # 如果遇到空行（或只有空白字符的行），则结束当前段落块
                if not text.strip():
                    if current_block:  # 如果当前块不为空
                        block_text = '\n'.join(current_block)
                        segment_info = {
                            'content': block_text,
                            'source_file': filename,
                            'file_path': file_path,
                            'doc_id': doc_id,  # 添加文档ID
                            'chunk_index': chunk_index,  # 添加段落块序号
                            'text_length': len(block_text)
                        }
                        segments.append(segment_info)
                        current_block = []
                        chunk_index += 1
                else:
                    # 将非空行添加到当前块
                    current_block.append(text)

            # 处理文档末尾的最后一个段落块
            if current_block:
                block_text = '\n'.join(current_block)
                segment_info = {
                    'content': block_text,
                    'source_file': filename,
                    'file_path': file_path,
                    'doc_id': doc_id,  # 添加文档ID
                    'chunk_index': chunk_index,  # 添加段落块序号
                    'text_length': len(block_text)
                }
                segments.append(segment_info)

        except Exception as e:
            logger.error(f"读取docx文件 {filename} 时出错: {str(e)}")
            raise

        return segments

    def _process_doc(self, file_path: str, filename: str) -> List[Dict]:
        logger.warning(f"DOC处理功能需要额外实现: {filename}")
        return []

    def _process_pdf(self, file_path: str, filename: str) -> List[Dict]:
        logger.warning(f"PDF处理功能需要额外实现: {filename}")
        return []

    def _process_txt(self, file_path: str, filename: str) -> List[Dict]:
        logger.warning(f"TXT处理功能需要额外实现: {filename}")
        return []


# 使用示例
if __name__ == "__main__":
    # 创建文档切分器实例
    splitter = DocumentSplitter(target_directory)

    # 切分所有文档
    segments = splitter.split_all_documents()

    # 打印结果示例
    for i, segment in enumerate(segments[:10]):  # 只显示前10个段落
        print(f"  文档名: {segment['source_file']}")
        print(f"  内容: {segment['content'][:100]}...")  # 显示前100个字符
        print(f"  doc_id: {segment['doc_id']} ")
        print(f"  chunk_index: {segment['chunk_index']} ")
        print("-" * 50)