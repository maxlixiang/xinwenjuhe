import asyncio
import json
import logging
from typing import List, Dict, Any

# 导入自定义类和异常（如果在不同文件，需调整导入路径）
# 假设 NewsFetcher/FetchError 在 news_fetcher.py，LLMNewsParser/ParseError 在 llm_news_parser.py
# 若所有类在同一文件，可删除以下导入，保留类定义即可
from news_fetcher import NewsFetcher, FetchError
from llm_news_parser import LLMNewsParser, ParseError

# 配置全局日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # 控制台输出
    ]
)
logger = logging.getLogger("news_intelligence")

# ------------------- 配置项 -------------------
# 替换为你的LLM API Key（必填）
LLM_API_KEY = "your-openai-api-key-here"
# 可选：自定义LLM基础地址（如智谱/通义千问）
LLM_BASE_URL = None  # 示例："https://open.bigmodel.cn/api/paas/v4/"
# LLM模型名称
LLM_MODEL = "gpt-3.5-turbo"

# 测试URL列表（1个正常新闻，1个404链接）
TEST_URLS = [
    # 正常新闻链接（网易新闻示例）
    "https://news.163.com/26/0220/10/IQ78V7F8000189FH.html",
    # 故意写错的404链接
    "https://news.163.com/26/0220/10/INVALID_404_PAGE.html"
]

# ------------------- 核心业务逻辑 -------------------
async def process_single_url(url: str, fetcher: NewsFetcher, parser: LLMNewsParser) -> None:
    """
    处理单个URL的抓取和解析流程
    :param url: 待处理的新闻URL
    :param fetcher: 已初始化的NewsFetcher实例
    :param parser: 已初始化的LLMNewsParser实例
    """
    logger.info(f"开始处理URL: {url}")
    try:
        # 1. 抓取HTML数据
        raw_html = await fetcher.fetch(url)
        logger.info(f"URL {url} 抓取成功，HTML长度: {len(raw_html)} 字节")
        
        # 2. 解析为结构化数据（需将bytes转为str）
        structured_data = await parser.parse(raw_html.decode("utf-8", errors="ignore"))
        
        # 3. 打印解析结果（格式化JSON）
        logger.info(f"URL {url} 解析成功，结果如下：")
        print(json.dumps(structured_data, ensure_ascii=False, indent=2))
        
    except FetchError as e:
        # 捕获抓取异常
        logger.error(f"URL {url} 抓取失败: {e} (失败原因: {e.reason})")
    except ParseError as e:
        # 捕获解析异常
        logger.error(f"URL {url} 解析失败: {e} (失败阶段: {e.step}, 原因: {e.reason})")
    except Exception as e:
        # 捕获未预期的异常（兜底）
        logger.error(f"URL {url} 处理时发生未预期错误: {str(e)[:200]}", exc_info=True)
    finally:
        logger.info(f"URL {url} 处理完成（无论成败）\n" + "-"*80)

async def main():
    """主函数：初始化组件 + 遍历处理所有测试URL"""
    # 1. 初始化LLM解析器（提前初始化，复用客户端）
    parser = LLMNewsParser(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        timeout=30.0,
        max_text_length=8000
    )
    
    # 2. 使用async with管理NewsFetcher资源
    async with NewsFetcher(
        timeout=10.0,
        max_retries=3,
        backoff_base=1.0,
        # 可选：添加内容校验函数
        content_validator=lambda content: len(content) > 100 and content.strip() != b""
    ) as fetcher:
        
        # 3. 并发处理所有 URL（榨干异步性能）
        logger.info(f"开始并发处理 {len(TEST_URLS)} 个URL...")
        
        # 创建一个任务列表
        tasks = [process_single_url(url, fetcher, parser) for url in TEST_URLS]
        
        # 使用 asyncio.gather 同时运行所有任务
        # return_exceptions=True 确保就算某个任务发生未捕获的严重异常，也不会导致其他任务被取消
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # 4. 日志收尾
    logger.info("所有URL处理完毕，程序正常退出")

# ------------------- 程序入口 -------------------
if __name__ == "__main__":
    # 检查必要配置
    if LLM_API_KEY == "your-openai-api-key-here":
        logger.error("请先替换 LLM_API_KEY 为你的有效API密钥！")
        exit(1)
    
    # 运行异步主函数
    asyncio.run(main())