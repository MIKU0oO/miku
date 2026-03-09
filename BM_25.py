import json
import jieba
import numpy as np
import torch
import os

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
EMBEDDING_SAVE_PATH = "embeddings.npy"


# 数据加载
def load_articles(jsonl_file, max_lines=10000):
    articles = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        num=0
        for i, line in enumerate(file):
            #num+=1
            #if num >100:
                #break
            if i >= max_lines:
                break
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return articles


# 构建 BM25
def build_bm25(articles):
    corpus = [list(jieba.cut(a['completion'])) for a in articles]
    bm25 = BM25Okapi(corpus)
    return bm25, corpus


def bm25_search(query, bm25, articles, top_k=50):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [articles[i] for i in top_indices]



# 构建 BGE Embedding
class EmbeddingRetriever:
    def __init__(self, model_name="BAAI/bge-large-zh", device="cpu"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def build_embeddings(self, articles):
        texts = [a['completion'] for a in articles]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )

        return embeddings

    def save_embeddings(self, embeddings, path=EMBEDDING_SAVE_PATH):
        np.save(path, embeddings)
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path=EMBEDDING_SAVE_PATH):
        embeddings = np.load(path)
        print(f"Embeddings loaded from {path}")
        return embeddings

    def search(self, query, embeddings, articles, top_k=50):
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        scores = np.dot(embeddings, query_vec.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [articles[i] for i in top_indices]


# BGE Reranker 精排
class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def rerank(self, query, candidates, top_n=5):
        pairs = [[query, c['completion']] for c in candidates]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        scores = scores.cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1][:top_n]
        return [candidates[i] for i in sorted_indices]


# 完整检索流程

class TwoStageRetriever:
    def __init__(self, jsonl_file):
        print("Loading articles...")
        self.articles = load_articles(jsonl_file)

        print("Building BM25...")
        self.bm25, _ = build_bm25(self.articles)

        print("Loading embedding model...")
        self.embedding_model = EmbeddingRetriever()

        if os.path.exists(EMBEDDING_SAVE_PATH):
            print("Loading existing embeddings...")
            self.embeddings = self.embedding_model.load_embeddings()
        else:
            print("Building embeddings...")
            self.embeddings = self.embedding_model.build_embeddings(self.articles)
            self.embedding_model.save_embeddings(self.embeddings)

        print("Loading reranker...")
        self.reranker = Reranker()

    def retrieve(self, query, top_k=50, top_n=5, use_bm25=False):
        # 第一阶段召回
        if use_bm25:
            candidates = bm25_search(query, self.bm25, self.articles, top_k)
        else:
            candidates = self.embedding_model.search(
                query,
                self.embeddings,
                self.articles,
                top_k
            )

        # 第二阶段精排
        final_docs = self.reranker.rerank(query, candidates, top_n)

        return final_docs



# 使用示例

if __name__ == "__main__":
    retriever = TwoStageRetriever("training_data.jsonl")

    query = "5月29日至31日，上海合作组织减贫和可持续发展论坛在哪举行?"

    results = retriever.retrieve(
        query,
        top_k=50,     # 第一阶段召回数量
        top_n=5,      # 最终返回数量
        use_bm25=False  # True=BM25 False=Embedding
    )

    for i, r in enumerate(results):
        print(f"\n===== Top {i+1} =====")
        print(r["completion"])