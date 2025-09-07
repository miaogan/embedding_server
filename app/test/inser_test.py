import pandas as pd
import requests
import json
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import time
import os

# --- 配置 ---
# 优先从环境变量读取配置
TSV_FILE_PATH = os.getenv("TSV_FILE_PATH",
                          "D:\\work_space\\python_project\\embedding_server\\quora_duplicate_questions.tsv")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hybrid_demo")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://xxx:5000/embed")  # FastAPI 服务地址
HEALTH_CHECK_URL = os.getenv("HEALTH_CHECK_URL", "http://xxxx:5000/health")  # FastAPI 健康检查地址
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))
MAX_DOCS = int(os.getenv("MAX_DOCS", 500))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 120))  # API 调用超时时间（秒）
API_RETRIES = int(os.getenv("API_RETRIES", 3))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))  # 重试前等待时间（秒）


def check_embedding_service_health(health_url, retries=3, delay=5):
    """检查嵌入服务是否健康"""
    for attempt in range(retries):
        try:
            print(f"检查嵌入服务健康状态 (尝试 {attempt + 1}/{retries})...")
            response = requests.get(health_url, timeout=10)
            response.raise_for_status()
            health_data = response.json()
            if health_data.get("status") == "healthy" and health_data.get("model_loaded") == True:
                print("嵌入服务健康检查通过。")
                return True
            else:
                print(f"嵌入服务未就绪: {health_data}")
        except requests.exceptions.RequestException as e:
            print(f"健康检查请求失败: {e}")
        except Exception as e:
            print(f"解析健康检查响应失败: {e}")

        if attempt < retries - 1:
            print(f"等待 {delay} 秒后重试...")
            time.sleep(delay)
    print("嵌入服务健康检查失败，达到最大重试次数。")
    return False


def load_data(file_path, max_docs):
    """从TSV文件加载数据"""
    print("开始加载数据...")
    try:
        df = pd.read_csv(file_path, sep="\t")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}。")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

    questions = set()
    for _, row in df.iterrows():
        obj = row.to_dict()
        questions.add(obj["question1"][:512])
        questions.add(obj["question2"][:512])
        if len(questions) >= max_docs:
            break

    docs = list(questions)
    print(f"共加载 {len(docs)} 个唯一问题。")
    return docs


def get_embeddings_via_api(texts, service_url, retries=API_RETRIES, delay=RETRY_DELAY):
    """通过FastAPI服务获取嵌入"""
    payload = {"texts": texts}
    headers = {'Content-Type': 'application/json'}

    for attempt in range(retries):
        try:
            print(f"尝试调用嵌入服务 (第 {attempt + 1}/{retries} 次)...")
            response = requests.post(service_url, data=json.dumps(payload), headers=headers, timeout=API_TIMEOUT)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                print("成功从API获取嵌入。")
                # FastAPI 返回的是嵌套的 data 字段
                return result.get("data")
            else:
                error_msg = result.get("error", "未知错误")
                print(f"API返回错误: {error_msg}")

        except requests.exceptions.Timeout:
            print(f"API 调用超时 ({API_TIMEOUT}秒)。")
        except requests.exceptions.RequestException as e:
            print(f"请求嵌入服务时出错: {e}")
        except json.JSONDecodeError:
            print("响应内容不是有效的JSON格式。")
        except Exception as e:
            print(f"解析API响应时出错: {e}")

        if attempt < retries - 1:
            print(f"等待 {delay} 秒后重试...")
            time.sleep(delay)
        else:
            print("达到最大重试次数，获取嵌入失败。")

    return None


def setup_milvus_collection(col_name, dense_dim):
    """设置Milvus集合"""
    print("开始连接 Milvus...")
    try:
        connections.connect(uri=MILVUS_URI)
        print("成功连接到 Milvus。")
    except Exception as e:
        print(f"连接 Milvus 时发生错误: {e}")
        return None

    if utility.has_collection(col_name):
        print(f"集合 '{col_name}' 已存在，正在删除...")
        try:
            Collection(col_name).drop()
            print(f"集合 '{col_name}' 已删除。")
        except Exception as e:
            print(f"删除集合时发生错误: {e}")
            return None

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields, description="用于混合搜索的演示集合")

    try:
        col = Collection(col_name, schema, consistency_level="Strong")
        print(f"集合 '{col_name}' 创建成功。")
    except Exception as e:
        print(f"创建集合时发生错误: {e}")
        return None

    try:
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        print("稀疏向量索引创建成功。")

        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)
        print("稠密向量索引创建成功。")
    except Exception as e:
        print(f"创建索引时发生错误: {e}")
        return None

    try:
        col.load()
        print(f"集合 '{col_name}' 已加载。")
    except Exception as e:
        print(f"加载集合时发生错误: {e}")
        return None

    return col


def insert_data_to_milvus(collection, docs, embedding_service_url, batch_size):
    """将数据和嵌入插入Milvus"""
    print("开始插入数据到 Milvus...")
    total_inserted = 0
    failed_batches = 0

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        print(f"处理批次 {i // batch_size + 1}: 文本 {i + 1} 到 {min(i + batch_size, len(docs))}")

        # 1. 调用API获取嵌入
        embeddings_data = get_embeddings_via_api(batch_docs, embedding_service_url)
        if not embeddings_data:
            print(f"批次 {i // batch_size + 1} 获取嵌入失败，跳过此批次。")
            failed_batches += 1
            continue

        # 2. 准备实体数据
        try:
            # FastAPI 返回的结构是 {"sparse": [...], "dense": [...]}
            batch_entities = [
                batch_docs,  # text
                embeddings_data["sparse"],  # sparse_vector
                embeddings_data["dense"]  # dense_vector
            ]
        except KeyError as e:
            print(f"API返回的嵌入数据缺少键 {e}，跳过此批次。")
            failed_batches += 1
            continue
        except Exception as e:
            print(f"准备实体数据时出错: {e}")
            failed_batches += 1
            continue

        # 3. 插入到Milvus
        try:
            insert_result = collection.insert(batch_entities)
            total_inserted += len(insert_result.primary_keys)
            print(f"批次 {i // batch_size + 1} 插入完成，当前共插入 {total_inserted} 个实体。")
        except Exception as e:
            print(f"批次 {i // batch_size + 1} 插入 Milvus 时出错: {e}")
            failed_batches += 1

    try:
        collection.flush()
        print("数据插入完成并已刷新。")
        final_count = collection.num_entities
        print(f"最终集合中的实体数量: {final_count}")
        print(f"总共处理了 {len(docs)} 个文档，成功插入 {total_inserted} 个，失败批次 {failed_batches} 个。")
        return final_count
    except Exception as e:
        print(f"刷新数据时出错: {e}")
        return -1


def main():
    """主函数"""
    # 0. 检查嵌入服务健康
    if not check_embedding_service_health(HEALTH_CHECK_URL):
        print("嵌入服务不可用，程序退出。")
        return

    # 1. 加载数据
    docs = load_data(TSV_FILE_PATH, MAX_DOCS)
    if not docs:
        print("没有加载到数据，程序退出。")
        return

    # 2. 获取一个示例嵌入以确定 dense vector 的维度
    print("正在获取向量维度信息...")
    dummy_embeddings = get_embeddings_via_api([docs[0]], EMBEDDING_SERVICE_URL)
    if not dummy_embeddings or 'dense' not in dummy_embeddings or not dummy_embeddings['dense']:
        print("无法获取向量维度，程序退出。")
        return
    try:
        # 假设 dense 向量是二维列表 [[...]]
        dense_dim = len(dummy_embeddings['dense'][0])
        print(f"确定稠密向量维度为: {dense_dim}")
    except (IndexError, TypeError, KeyError) as e:
        print(f"无法从示例嵌入中确定维度: {e}")
        return

    # 3. 设置 Milvus 集合
    collection = setup_milvus_collection(COLLECTION_NAME, dense_dim)
    if not collection:
        print("Milvus 集合设置失败，程序退出。")
        return

    # 4. 插入数据
    insert_data_to_milvus(collection, docs, EMBEDDING_SERVICE_URL, BATCH_SIZE)

    print("Milvus 客户端脚本执行完毕。")


if __name__ == "__main__":
    main()



