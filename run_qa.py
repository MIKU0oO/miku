from BM_25 import TwoStageRetriever
from train_data import load_prompts
REQ_TIME_GAP = 1
import random
import time
import requests

key ="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
url = "https://xiaoai.plus/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
}

def get_openai_response(query,prompts):
    try:
        payload = {
            "model": "gpt-4o-2024-11-20",
            "messages" : [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "你是一个问答系统助手，会根据我给的新闻信息回答问题。\n"
                            "简要的给出答案，无需解释。"
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"以下为新闻内容：\n{prompts}\n\n"
                            f"请你回答问题：\n{query}\n\n"
                        )
                    }
                ]
            }
        ],
            "temperature": 0.4,
            "max_tokens": 300
        }
        #print(example_text,image_url)
        response = requests.post(url, headers=headers, json=payload,timeout=60)
        time.sleep(random.randint(1, REQ_TIME_GAP))
        if response.status_code == 200:
            result = response.json()
            text=result["choices"][0]["message"]["content"]
            return text
        else:
            print("请求失败：", response.status_code, response.text)
            return 0

    except Exception as e:
        print(e)
        time.sleep(REQ_TIME_GAP)
        print("其他错误")
    return 0


querys=["2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？",
"2024年是中国红十字会成立多少周年",
"2024年我国文化和旅游部部长是谁？",
"《中华人民共和国爱国主义教育法》什么时候实施？",
"2023年全国电影总票房为多少元？",
"蒙古人民党总书记是谁？",
"2023—2024赛季国际滑联短道速滑世界杯北京站比赛中，刘少昂参与获得几枚奖牌？",
"福建自贸试验区在自贸建设十年中主要从哪几个方面推动改革创新？",
"杭州第十九届亚洲运动会共举行多少天？",
"哪些单位在中国期刊高质量发展论坛的主论坛上做主题演讲？"]

# 示例使用
if __name__ == "__main__":
    query=querys[0]
    retriever = TwoStageRetriever("training_data.jsonl")
    prompts = retriever.retrieve(
        query,
        top_k=50,  # 第一阶段召回数量
        top_n=5,  # 最终返回数量
        use_bm25=False  # True=BM25 False=Embedding
    )
    #prompts = load_prompts(query)
    print(prompts)
    answer = get_openai_response(query, prompts)
    print(f"Query: {query}")
    print(f"Answer: {answer}")




def evaluate(queries: list):

    return [
        "湖南第一师范学院",
        "120",
        "孙业礼",
        "2024年1月1日",
        "549.15亿",
        "阿玛尔巴伊斯格楞",
        "2",
        "推进制度集成创新；服务海峡两岸融合发展；深化共建“一带一路”",
        "16",
        "中国科协科技创新部、湖南省委宣传部、上海大学、《历史研究》、《读者》、《分子植物》、《问天少年》、南方杂志社、中华医学会杂志社",
    ] 
