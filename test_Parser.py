import re
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Coroutine, Any, Dict, Optional
from bs4 import BeautifulSoup, Comment
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError

# 基础抽象类
class BaseParser(ABC):
    @abstractmethod
    async def parse(self, raw_data: str) -> Coroutine[Any, Any, Dict[str, Any]]:
        pass

# 自定义解析异常
class ParseError(Exception):
    """自定义的新闻解析异常，标识HTML清理或LLM解析失败"""
    def __init__(self, step: str, reason: str):
        self.step = step  # 失败阶段：html_clean / llm_request / json_parse / not_news
        self.reason = reason
        super().__init__(f"解析失败（{step}）: {reason}")

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_news_parser")

class LLMNewsParser(BaseParser):
    """
    基于LLM的新闻解析器（重构版）：修复正则Bug、Token截断、防幻觉、原生JSON Mode
    兼容OpenAI接口的大模型（OpenAI/智谱/通义千问等）
    """
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_text_length: int = 8000  # 新增：文本截断长度配置
    ):
        # 初始化OpenAI异步客户端
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_text_length = max_text_length  # 文本最大长度

    def _clean_html(self, raw_html: str) -> str:
        """
        重构点1：修复HTML清理的正则Bug，保留换行符
        清理HTML，提取纯净文本（剔除无关标签，仅压缩空格/制表符，保留换行）
        """
        try:
            # 初始化BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            
            # 移除无关标签：script、style、noscript、注释等
            for tag in soup(["script", "style", "noscript", "iframe", "header", "footer"]):
                tag.decompose()
            # 移除HTML注释
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # 提取文本并清理空白字符
            clean_text = soup.get_text(separator="\n")
            
            # 重构点1修复：仅压缩空格和制表符，保留换行符
            # 步骤1：替换多个空格/制表符为单个空格
            clean_text = re.sub(r"[ \t]+", " ", clean_text)
            # 步骤2：替换多个连续换行符为单个换行（可选，保持段落结构）
            clean_text = re.sub(r"\n+", "\n", clean_text)
            # 步骤3：移除首尾空白
            clean_text = clean_text.strip()
            
            if not clean_text:
                raise ParseError("html_clean", "HTML清理后无有效文本")
            
            return clean_text
        
        except Exception as e:
            raise ParseError("html_clean", f"HTML解析失败: {str(e)[:200]}")

    def _clean_llm_json(self, llm_response: str) -> str:
        """保留原有JSON清洗逻辑，作为原生JSON Mode的兜底"""
        # 移除Markdown的```json和```标记
        json_pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(json_pattern, llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 找到第一个{和最后一个}之间的内容
        start_idx = llm_response.find("{")
        end_idx = llm_response.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return llm_response[start_idx:end_idx+1].strip()
        
        return llm_response.strip()

    async def _call_llm(self, clean_text: str) -> Dict[str, Any]:
        """
        重构点2/3/4：文本截断、优化Prompt防幻觉、启用原生JSON Mode
        调用LLM提取结构化信息
        """
        # 重构点2：文本截断，防止Token爆炸（仅保留前max_text_length个字符）
        if len(clean_text) > self.max_text_length:
            logger.warning(f"文本长度({len(clean_text)})超出限制，截断至{self.max_text_length}字符")
            clean_text = clean_text[:self.max_text_length]

        # 重构点3：优化Prompt，防止模型幻觉（非新闻页面返回全null）
        system_prompt = """
你是一个专业的新闻解析助手，需要严格遵守以下规则处理文本：
1. 仅从给定文本中提取信息，绝对不要编造任何数据；
2. 如果判断传入的文本根本不是一篇新闻报道（例如是登录提示、404错误、验证码页面、广告页面、空白页面等），请让所有字段的值严格返回null；
3. 若为有效新闻，提取以下字段并返回JSON格式：
   - title: 新闻标题（字符串，无则为null）
   - content: 新闻正文内容（字符串，保留核心信息，无则为null）
   - publish_time: 发布时间（ISO 8601格式，如2026-02-20T14:30:00+08:00，无则为null）
   - author: 作者（字符串，无则为null）
4. 严格按照上述字段返回，不要新增/遗漏字段，确保JSON格式完全合法，无任何额外解释或文本。
        """.strip()

        user_prompt = f"请解析以下文本，严格按要求返回JSON：\n{clean_text}"

        try:
            # 重构点4：启用原生JSON Mode，提升输出稳定性
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                timeout=self.timeout,
                response_format={"type": "json_object"}  # 原生JSON模式
            )

            # 提取LLM返回内容
            llm_output = response.choices[0].message.content.strip()
            if not llm_output:
                raise ParseError("llm_request", "LLM返回空内容")


            
            # 解析JSON
            try:
                structured_data = json.loads(llm_output)
            except json.JSONDecodeError as e:
                raise ParseError(
                    "json_parse",
                    f"JSON解析失败（原始输出：{llm_output[:200]}，清洗后：{clean_json_str[:200]}）：{str(e)}"
                )

            # 校验必填字段
            required_fields = ["title", "content", "publish_time", "author"]
            missing_fields = [f for f in required_fields if f not in structured_data]
            if missing_fields:
                raise ParseError(
                    "json_parse",
                    f"JSON缺少必填字段：{missing_fields}，解析结果：{structured_data}"
                )

            # 重构点3：业务逻辑校验，判断是否为有效新闻
            title = structured_data.get("title")
            content = structured_data.get("content")
            if title is None and content is None:
                raise ParseError("not_news", "该页面不是有效的新闻内容（无标题和正文）")

            return structured_data

        except APITimeoutError:
            raise ParseError("llm_request", f"LLM请求超时（{self.timeout}秒）")
        except APIConnectionError:
            raise ParseError("llm_request", "LLM API连接失败（网络/地址错误）")
        except APIError as e:
            raise ParseError("llm_request", f"LLM API错误：{e.message}")
        except ParseError:
            raise
        except Exception as e:
            raise ParseError("llm_request", f"LLM交互未知错误：{str(e)[:200]}")

    async def parse(self, raw_data: str) -> Dict[str, Any]:
        """核心解析方法：HTML清理 → LLM提取（重构后）"""
        try:
            # 第一步：清理HTML提取纯文本
            logger.debug("开始清理HTML文本（保留段落结构）")
            clean_text = self._clean_html(raw_data)
            
            # 第二步：调用LLM提取结构化信息
            logger.debug("调用LLM提取新闻结构化信息（启用JSON Mode）")
            structured_data = await self._call_llm(clean_text)
            
            return structured_data

        except ParseError as e:
            logger.error(f"解析失败：{e}")
            raise
        except Exception as e:
            error_msg = f"解析未预期错误：{str(e)[:200]}"
            logger.error(error_msg)
            raise ParseError("unknown", error_msg)

    async def close(self):
        """关闭客户端连接（预留扩展）"""
        pass

# ------------------- 测试用例 -------------------
async def test_llm_news_parser():
    """测试重构后的LLMNewsParser"""
    # 替换为你的API Key
    API_KEY = "your-api-key-here"
    
    # 测试1：有效新闻HTML
    valid_news_html = """
    <html>
        <head><title>2026年人工智能发展报告发布</title></head>
        <body>
            <script>alert('广告')</script>
            <div class="content">
                <h1>2026年人工智能发展报告发布</h1>
                <p>作者：科技日报记者 李明</p>
                <p>发布时间：2026-02-20 09:30:00</p>
                <p>近日，工信部发布了《2026年人工智能产业发展报告》，报告指出...</p>
                <p>人工智能在制造业的渗透率已达35%，较去年提升5个百分点。</p>
            </div>
        </body>
    </html>
    """

    # 测试2：非新闻页面（404）
    non_news_html = """
    <html>
        <head><title>404 Not Found</title></head>
        <body>
            <h1>404 - 页面不存在</h1>
            <p>你访问的页面已被删除或不存在，请检查URL是否正确。</p>
            <p>请返回<a href="/">首页</a>继续浏览</p>
        </body>
    </html>
    """

    try:
        # 初始化解析器
        parser = LLMNewsParser(
            api_key=API_KEY,
            model="gpt-3.5-turbo",
            timeout=30.0,
            max_text_length=8000  # 可自定义截断长度
        )
        
        # 测试有效新闻
        print("=== 测试有效新闻 ===")
        result = await parser.parse(valid_news_html)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 测试非新闻页面（会抛出not_news异常）
        print("\n=== 测试非新闻页面 ===")
        result = await parser.parse(non_news_html)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except ParseError as e:
        print(f"\n解析失败：{e}（阶段：{e.step}）")

if __name__ == "__main__":
    asyncio.run(test_llm_news_parser())