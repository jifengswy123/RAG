import uuid

# 为doc文件生成uuid用于doc_id
def generate_doc_id_by_hash(filename):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS,filename))
