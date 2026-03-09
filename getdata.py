#!/usr/bin/python3
# -*- coding:utf-8 -*-
import requests
import re
import json
from lxml import etree
from retry import retry

articles=[]

@retry(retry=3,sleep=5)
def get_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url,headers=headers)
    response.encoding = "utf-8"
    if response.status_code == 200:
        return response.text
    else:
        print(response.status_code)
        return "ERROR"

def get_text(text):
    if isinstance(text,str):
        return re.sub("\\r|\\n|\\t|　| ", "", text).strip(" ")
    elif isinstance(text,list):
        return "".join([re.sub("\\r|\\n|\\t|　| ", "", i).strip(" ") for i in text])


def anlise_detail(detail_html):
    global articles

    tree = etree.HTML(detail_html)
    lis = tree.xpath('//div[@class="article"]|//div[@class="text_c"]')
    for li in lis:
        title = get_text(li.xpath('./h1/text()'))
        #print("标题",title)
        title2 = get_text(li.xpath('./h2/text()')).strip("\n")
        #if title2:
            #print("副标题",title2)
        pusblish_info = get_text(li.xpath('.//span[@class="date"]/text()|//div[@class="lai"]//text()'))
        #print("文章信息",pusblish_info)
        content = get_text(li.xpath('.//div[@id="ozoom"]//p/text()'))
        #print(content)

        #构建字典
        article_data= {
            "title":title,
            'url':detail_url,
            "pusblish_info":pusblish_info,
            "content":content
        }
        articles.append(article_data)


year_list = [str(i) for i in range(2023,2024)]
month_list = [str(i).zfill(2) for i in range(1,13)]
day_list = [str(i).zfill(2) for i in range(1,32)]

for year in year_list:
    for month in month_list:
        for day in day_list:
            head = "http://paper.people.com.cn/rmrb/html/{}-{}/{}/".format(year,month,day)
            for i in range(1,21):
                url = "{}nbs.D110000renmrb_{}.htm".format(head,str(i).zfill(2))
                #print(url)
                html = get_html(url)
                if html != "ERROR":
                    tree = etree.HTML(html)
                    lis = tree.xpath('//div[@class="news"]/ul|//div[@id="titleList"]/ul')
                    for li in lis:
                        detail_url_list = li.xpath('./li/a/@href')
                        name_list = li.xpath('./li/a//text()')
                        for name,_url in zip(name_list, detail_url_list):
                            detail_url = "{}{}".format(head,_url)
                            #name = re.findall('document\.write\(view\(\"(.*?)\"\)\)',name)[0].strip()
                            print(name,detail_url)
                            detail_html = get_html(detail_url)
                            if detail_html != "ERROR":
                                anlise_detail(detail_html)

with open("articles.json","w",encoding="utf-8") as f:
    json.dump(articles,f,ensure_ascii=False,indent=2)