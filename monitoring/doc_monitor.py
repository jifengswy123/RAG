import time
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import docx
from collections import defaultdict
from util.doc_Splitter import DocumentSplitter
from util.util import generate_doc_id_by_hash
from util.useEmbeddingModel import get_embedding
from util.connectMilvus import validate_insert_data,insert_milvus,delete_milvus
from pymilvus import FieldSchema, DataType
from config import config


class DocFileHandler(FileSystemEventHandler):
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        # 存储当前目录下已有的文件
        self.existing_files = set(os.listdir(watch_dir) if os.path.exists(watch_dir) else [])
        # 用于跟踪处理中的文件，防止重复处理
        self.processing_files = set()
        # 用于跟踪文件的最后修改时间
        self.file_timestamps = {}
        # 用于延迟处理
        self.event_queue = defaultdict(list)
        self.lock = threading.Lock()

    def _read_doc_content(self, filepath):
        """读取docx文档内容"""
        try:
            doc = docx.Document(filepath)
            content = []
            for para in doc.paragraphs:
                content.append(para.text)
            # return "\n".join(content) if content else "[空文档]"
            return content if content else "[空文档]"
        except Exception as e:
            return f"读取文档内容失败: {e}"

    def _process_event_later(self, event_type, filepath, delay=0.5):
        """延迟处理事件，避免重复触发"""
        filename = os.path.basename(filepath)

        with self.lock:
            if filename in self.processing_files:
                return False
            self.processing_files.add(filename)

        def delayed_process():
            time.sleep(delay)

            try:
                if event_type == "created":
                    self._handle_created_event(filepath)
                elif event_type == "modified":
                    self._handle_modified_event(filepath)

            finally:
                with self.lock:
                    if filename in self.processing_files:
                        self.processing_files.remove(filename)

        thread = threading.Thread(target=delayed_process, daemon=True)
        thread.start()
        return True

    def _handle_created_event(self, filepath):
        """处理创建事件"""
        if not os.path.exists(filepath):
            return

        filename = os.path.basename(filepath)

        # 获取文件的修改时间
        current_mtime = os.path.getmtime(filepath)

        with self.lock:
            # 检查是否是真正的覆盖操作
            if filename in self.existing_files:
                # 这是覆盖操作，由on_modified处理
                return

            print(f"[新增文档] 文件名: {filename}")
            print(f"开始处理文档{filepath}")
            # 1.获取title
            title = filename
            # 2.获取doc_id
            doc_id = generate_doc_id_by_hash(filename)
            # 3.获取content和chunk_index
            try:
                splitter = DocumentSplitter()
                result = splitter.split_docx(filepath)
                print("打印元素：")
                print(result[0][0])
            except FileNotFoundError:
                print(f"错误: 文件 '{filepath}' 不存在")
            except Exception as e:
                print(f"错误: {e}")

            # 组装新的List并将content向量化为vector
            milvuslist = []
            for i in range(len(result)):
                content = result[i][0]
                chunk_index = result[i][1]
                vector = get_embedding(content)
                dict = {"title": filename, "doc_id": doc_id, "content": content, "chunk_index": chunk_index, "vector": vector}
                milvuslist.append(dict)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            if validate_insert_data(milvuslist,fields):
                insert_milvus(milvuslist)
            print(f"文档处理完毕")
            print("打印milvuslist")
            print(milvuslist)
            milvuslist.clear()
            print("清空milvuslist")
            print(milvuslist)

            # 读取并打印内容
            try:
                content = self._read_doc_content(filepath)
            except Exception as e:
                print(f"读取文档内容时出错: {e}")

            # 更新状态
            self.existing_files.add(filename)
            self.file_timestamps[filename] = current_mtime
            print("-" * 50)


    def _handle_modified_event(self, filepath):
        """处理修改事件"""
        if not os.path.exists(filepath):
            return

        filename = os.path.basename(filepath)
        current_mtime = os.path.getmtime(filepath)
        doc_id = generate_doc_id_by_hash(filename)
        print(f"开始删除Milvus中旧文档{filename}内容")
        delete_milvus(doc_id)
        print("删除完成")
        # 1.获取title
        title = filename
        # 2.获取doc_id
        doc_id = generate_doc_id_by_hash(filename)
        # 3.获取content和chunk_index
        try:
            splitter = DocumentSplitter()
            result = splitter.split_docx(filepath)
            print("打印元素：")
            print(result[0][0])
        except FileNotFoundError:
            print(f"错误: 文件 '{filepath}' 不存在")
        except Exception as e:
            print(f"错误: {e}")

        milvuslist = []
        for i in range(len(result)):
            content = result[i][0]
            chunk_index = result[i][1]
            vector = get_embedding(content)
            dict = {"title": filename, "doc_id": doc_id, "content": content, "chunk_index": chunk_index,
                    "vector": vector}
            milvuslist.append(dict)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        if validate_insert_data(milvuslist, fields):
            insert_milvus(milvuslist)
        print(f"文档处理完毕")
        print("打印milvuslist")
        print(milvuslist)
        milvuslist.clear()
        print("清空milvuslist")
        print(milvuslist)

        with self.lock:
            # 只有当文件之前就存在，并且修改时间发生变化时，才认为是覆盖
            if filename in self.existing_files and filename in self.file_timestamps:
                previous_mtime = self.file_timestamps[filename]
                if current_mtime != previous_mtime:
                    print(f"[文档覆盖] 文件名: {filename}")

                    # 读取新内容
                    try:
                        content = self._read_doc_content(filepath)
                        print(f"新文档内容:\n{content}")
                    except Exception as e:
                        print(f"读取新文档内容时出错: {e}")

                    self.file_timestamps[filename] = current_mtime
                    print("-" * 50)
            else:
                # 文件不存在于记录中，说明是新增（可能在on_created中已处理）
                self.file_timestamps[filename] = current_mtime

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.doc', '.docx')):
            self._process_event_later("created", event.src_path, delay=0.5)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(('.doc', '.docx')):
            filename = os.path.basename(event.src_path)
            print(f"[删除文档] 文件名: {filename}")
            doc_id = generate_doc_id_by_hash(filename)
            print(f"开始删除Milvus中旧文档{filename}内容")
            delete_milvus(doc_id)
            print("删除完成")

            with self.lock:
                if filename in self.existing_files:
                    self.existing_files.remove(filename)
                if filename in self.file_timestamps:
                    del self.file_timestamps[filename]
            print("-" * 50)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.doc', '.docx')):
            # 忽略临时文件
            if event.src_path.startswith('~$'):
                return
            self._process_event_later("modified", event.src_path, delay=0.5)

    def on_moved(self, event):
        if not event.is_directory:
            src_is_doc = event.src_path and event.src_path.endswith(('.doc', '.docx'))
            dest_is_doc = event.dest_path and event.dest_path.endswith(('.doc', '.docx'))

            if src_is_doc or dest_is_doc:
                # 原文件被删除
                if src_is_doc:
                    src_name = os.path.basename(event.src_path)
                    with self.lock:
                        if src_name in self.existing_files:
                            self.existing_files.remove(src_name)
                        if src_name in self.file_timestamps:
                            del self.file_timestamps[src_name]

                # 新文件被创建
                if dest_is_doc and event.dest_path:
                    dest_name = os.path.basename(event.dest_path)

                    def process_moved():
                        time.sleep(0.5)
                        if os.path.exists(event.dest_path):
                            print(
                                f"[文件移动/重命名] 从: {os.path.basename(event.src_path) if event.src_path else '未知'} 到: {dest_name}")

                            try:
                                content = self._read_doc_content(event.dest_path)
                                print(f"文档内容:\n{content}")
                            except Exception as e:
                                print(f"读取文档内容时出错: {e}")

                            with self.lock:
                                self.existing_files.add(dest_name)
                                self.file_timestamps[dest_name] = os.path.getmtime(event.dest_path)
                            print("-" * 50)

                    thread = threading.Thread(target=process_moved, daemon=True)
                    thread.start()


def monitor_doc_directory(directory_path):
    """监控指定目录的Doc文档变化"""

    if not os.path.exists(directory_path):
        print(f"错误: 目录 {directory_path} 不存在")
        return

    if not os.path.isdir(directory_path):
        print(f"错误: {directory_path} 不是目录")
        return

    print(f"开始监控目录: {directory_path}")
    print(f"支持的文件类型: .doc, .docx")
    print("=" * 50)

    event_handler = DocFileHandler(directory_path)
    observer = Observer()
    observer.schedule(event_handler, directory_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止监控...")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    target_directory = config.target_directory

    try:
        monitor_doc_directory(target_directory)
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请先安装所需库: pip install watchdog python-docx")
    except Exception as e:
        print(f"监控过程中出错: {e}")