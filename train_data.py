import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba


def load_articles(jsonl_file):
    articles = []
    i=0
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            if i>10000:
                break
            i+=1
            try:
                articles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return articles


def jieba_tokenizer(text):

    return jieba.cut(text)

def find_most_similar_articles(query, articles, top_n=5):

    contents = [article['completion'] for article in articles]
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
    tfidf_matrix = vectorizer.fit_transform(contents)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec,tfidf_matrix).flatten()
    indices = similarity.argsort()[-top_n:][::-1]
    return [articles[i] for i in indices]


def load_prompts(query):
    """
    创建用于 API 调用的 prompt。
    """
    articles= load_articles('training_data.jsonl')
    similar_articles = find_most_similar_articles(query, articles, 5)
    prompt=""
    for article in similar_articles:
        prompt+=(article['completion']+"\n")
    return prompt

'''
# Test
query="深圳国际仲裁院创设于哪一年?"
print(load_prompts(query))
'''