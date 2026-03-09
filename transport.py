import json


def convert_articles_to_jsonl(input_file, output_file):
    """
    将 articles.json 文件转换为 training_data.jsonl 文件。
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        articles = json.load(infile)

        for article in articles:
            prompt = f"标题: {article['title']}\nURL: {article['url']}\n发布时间: {article['pusblish_info']}\n\n"
            completion = f"内容: {article['content']}"
            json_line = json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)
            outfile.write(json_line + "\n")


# 使用示例
convert_articles_to_jsonl('articles.json', 'training_data.jsonl')
