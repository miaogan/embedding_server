from pymilvus.model.hybrid import BGEM3EmbeddingFunction

query = input("Enter your search query: ")
print(query)
ef = BGEM3EmbeddingFunction(
    model_name='E:\\bge-m3',  # Specify the model name
    device='cuda:0',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)
query_embeddings = ef([query])
print(query_embeddings)
