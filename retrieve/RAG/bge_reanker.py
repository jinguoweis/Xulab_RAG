# 使用 BGE 模型进行重排序
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 使用 RRF 进行分数融合
# rrf_results = rrf_fusion(bm25_topk_matches, faiss_topk_matches, k=60)

# 获取融合后 top 3 的文档
# top3_docs = [docs[idx].page_content for idx, _ in rrf_results[:3]]
def bge_rerank(query, docs):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("/home/extra1T/model_embeding/Xorbits/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("/home/extra1T/model_embeding/Xorbits/bge-reranker-base")
    
    # 预处理
    inputs = tokenizer([query] * len(docs), docs, padding=True, truncation=True, return_tensors="pt",max_length=512)
    
    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取重排序得分（logits），并按得分从高到低排序
    scores = outputs.logits.squeeze().tolist()
    reranked_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    return reranked_results

# 使用 BGE 模型对 RRF 融合后的 top3 进行重排序
# reranked_results = bge_rerank(query, top3_docs)

# 输出重排序后的 top 3 结果
# print("重排序后的 Top 3 结果：")
# for doc, score in reranked_results:
#     print(f"文档: {doc}, 得分: {score}")
