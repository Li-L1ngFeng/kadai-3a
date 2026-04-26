import os
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"

from icrawler.builtin import BingImageCrawler

# 使用 BingImageCrawler 替代 GoogleImageCrawler（在代理环境下通常更稳定）
crawler = BingImageCrawler(storage={"root_dir":"imgdata/kadai3a"}, downloader_threads=5)
crawler.crawl(keyword="kojima", max_num=12)