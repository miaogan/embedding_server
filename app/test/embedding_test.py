import pandas as pd
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

def main():
    # --- 1. 数据加载与预处理 ---
    print("开始加载数据...")
    file_path = "D:\\work_space\\python_project\\embedding_server\\quora_duplicate_questions.tsv"
    try:
        df = pd.read_csv(file_path, sep="\t")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    questions = set()
    for _, row in df.iterrows():
        obj = row.to_dict()
        # 限制文本长度以符合 Milvus 字段定义
        questions.add(obj["question1"][:512])
        questions.add(obj["question2"][:512])
        # 限制文档数量以加快处理速度（可选）
        if len(questions) > 500:
            break

    docs = list(questions)
    print(f"共加载 {len(docs)} 个唯一问题。")

    if not docs:
        print("没有加载到任何文档，程序退出。")
        return

    # --- 2. 文本嵌入生成 ---
    print("开始生成嵌入向量...")
    try:
        # 确保模型路径 'E:\\bge-m3' 正确且模型已存在
        ef = BGEM3EmbeddingFunction(
            model_name='E:\\bge-m3',
            device='cpu',  # 如果没有GPU或遇到问题，可以先用 'cpu'
            use_fp16=False
        )
        docs_embeddings = ef(docs)
        dense_dim = ef.dim["dense"]
        print("嵌入向量生成完成。")
    except Exception as e:
        print(f"生成嵌入向量时发生错误: {e}")
        print("请检查模型路径和模型文件是否正确。")
        return

    # --- 3. Milvus 连接与集合设置 ---
    print("开始连接 Milvus 并设置集合...")
    try:
        connections.connect(uri="http://localhost:19530")
        print("成功连接到 Milvus。")
    except Exception as e:
        print(f"连接 Milvus 时发生错误: {e}")
        print("请确保 Milvus 服务已在 http://localhost:19530 启动。")
        return

    col_name = "hybrid_demo"
    # 如果集合已存在，则删除
    if utility.has_collection(col_name):
        print(f"集合 '{col_name}' 已存在，正在删除...")
        try:
            Collection(col_name).drop()
            print(f"集合 '{col_name}' 已删除。")
        except Exception as e:
            print(f"删除集合时发生错误: {e}")
            return

    # 定义集合 Schema
    fields = [
        FieldSchema(
            name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields, description="用于混合搜索的演示集合")

    # 创建集合
    try:
        col = Collection(col_name, schema, consistency_level="Strong")
        print(f"集合 '{col_name}' 创建成功。")
    except Exception as e:
        print(f"创建集合时发生错误: {e}")
        return

    # --- 4. 创建索引 ---
    print("开始为向量字段创建索引...")
    try:
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        col.create_index("sparse_vector", sparse_index)
        print("稀疏向量索引创建成功。")

        dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
        col.create_index("dense_vector", dense_index)
        print("稠密向量索引创建成功。")
    except Exception as e:
        print(f"创建索引时发生错误: {e}")
        return

    # --- 5. 数据插入 ---
    print("开始将数据插入集合...")
    try:
        col.load() # 加载集合以准备插入
        total_inserted = 0
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            end_idx = min(i + batch_size, len(docs))
            batched_entities = [
                docs[i:end_idx],  # text
                docs_embeddings["sparse"][i:end_idx],  # sparse_vector
                docs_embeddings["dense"][i:end_idx],  # dense_vector
            ]
            insert_result = col.insert(batched_entities)
            total_inserted += len(insert_result.primary_keys)
            print(f"已插入 {total_inserted}/{len(docs)} 个实体...")

        col.flush() # 确保所有数据写入磁盘
        print("数据插入完成并已刷新。")
        print("最终插入的实体数量:", col.num_entities)
    except Exception as e:
        print(f"插入数据时发生错误: {e}")
        return

    print("脚本执行完毕。")

if __name__ == "__main__":
    main()



