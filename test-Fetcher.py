import asyncio
import logging
import random
from typing import Coroutine, Any, Callable, Optional
from abc import ABC, abstractmethod
import httpx

# 基础抽象类（如需拆分文件，可单独导入）
class BaseFetcher(ABC):
    @abstractmethod
    async def fetch(self, source_url: str) -> Coroutine[Any, Any, bytes | str]:
        pass

# 自定义抓取异常
class FetchError(Exception):
    """自定义的新闻抓取异常，用于标识抓取过程中的失败"""
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"抓取URL失败 {url}: {reason}")

# 自定义内容校验异常
class ContentValidationError(FetchError):
    """内容校验失败的专属异常，继承自FetchError"""
    def __init__(self, url: str, reason: str):
        super().__init__(url, f"内容校验失败: {reason}")

# 配置日志（保持简洁，级别可外部调整）
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("news_fetcher")

# 常见的浏览器User-Agent列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
]

class NewsFetcher(BaseFetcher):
    """
    基于httpx的异步新闻抓取器
    支持异步上下文管理器、可配置重试/超时、内容校验、优化日志
    """
    def __init__(
        self,
        timeout: float = 10.0,          # 超时时间（默认10秒）
        max_retries: int = 3,           # 最大重试次数（默认3次）
        backoff_base: float = 1.0,      # 退避基数（默认1秒）
        content_validator: Optional[Callable[[bytes | str], bool]] = None  # 内容校验函数
    ):
        # 解耦配置参数，设置默认值
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.content_validator = content_validator
        
        # 初始化httpx异步客户端
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True
        )

    # 实现异步上下文管理器入口
    async def __aenter__(self) -> "NewsFetcher":
        return self

    # 实现异步上下文管理器退出逻辑
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # 无论是否发生异常，都关闭客户端释放资源
        await self.client.aclose()

    async def fetch(self, source_url: str) -> bytes | str:
        """
        异步抓取指定URL的原始数据，包含重试、校验、优化日志逻辑
        
        Args:
            source_url: 新闻源URL
            
        Returns:
            抓取并校验通过的原始数据（bytes或str）
            
        Raises:
            FetchError: 所有重试失败时抛出
            ContentValidationError: 内容校验失败时抛出
        """
        last_error_reason = ""  # 记录最后一次失败的原因
        
        for attempt in range(self.max_retries):
            try:
                # 构建随机User-Agent请求头
                headers = {
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
                }
                
                # 发送异步请求
                response = await self.client.get(
                    url=source_url,
                    headers=headers
                )
                response.raise_for_status()  # 检查HTTP状态码
                
                # 获取原始内容（bytes类型，如需字符串可自行转换）
                content = response.content
                
                # 执行内容校验（如果配置了校验函数）
                if self.content_validator is not None:
                    if not self.content_validator(content):
                        raise ContentValidationError(
                            url=source_url,
                            reason=f"内容未通过校验（长度: {len(content)} 字节）"
                        )
                
                # 校验通过，返回内容
                return content
                
            except ContentValidationError:
                # 内容校验失败直接抛出，不重试（软404无需重试）
                raise
            except httpx.HTTPStatusError as e:
                last_error_reason = f"HTTP状态码错误: {e.response.status_code}"
                logger.warning(
                    f"抓取尝试 {attempt+1}/{self.max_retries} 失败 - {source_url}: {last_error_reason}"
                )
            except httpx.TimeoutException:
                last_error_reason = "请求超时"
                logger.warning(
                    f"抓取尝试 {attempt+1}/{self.max_retries} 失败 - {source_url}: {last_error_reason}"
                )
            except httpx.ConnectError:
                last_error_reason = "连接失败（DNS/拒绝连接）"
                logger.warning(
                    f"抓取尝试 {attempt+1}/{self.max_retries} 失败 - {source_url}: {last_error_reason}"
                )
            except Exception as e:
                last_error_reason = f"未知错误: {str(e)[:200]}"
                logger.warning(
                    f"抓取尝试 {attempt+1}/{self.max_retries} 失败 - {source_url}: {last_error_reason}"
                )
            
            # 指数退避重试（非最后一次尝试）
            if attempt < self.max_retries - 1:
                backoff_time = self.backoff_base * (2 ** attempt)
                logger.warning(
                    f"等待 {backoff_time:.1f} 秒后进行第 {attempt+2} 次重试..."
                )
                await asyncio.sleep(backoff_time)
        
        # 所有重试失败，记录ERROR级别日志并抛异常
        logger.error(f"所有 {self.max_retries} 次抓取尝试均失败 - {source_url}: {last_error_reason}")
        raise FetchError(url=source_url, reason=last_error_reason)

# ------------------- 测试用例 -------------------
async def test_news_fetcher():
    """测试优化后的NewsFetcher"""
    # 定义内容校验函数：长度大于100字节且非空
    def validate_content(content: bytes) -> bool:
        return len(content) > 100 and content.strip() != b""

    # 使用async with上下文管理器
    async with NewsFetcher(
        timeout=10.0,
        max_retries=3,
        backoff_base=1.0,
        content_validator=validate_content
    ) as fetcher:
        try:
            # 测试有效URL
            data = await fetcher.fetch("https://www.baidu.com")
            print(f"抓取成功，数据长度: {len(data)}")
            
            # 测试无效URL（触发重试和ERROR日志）
            # data = await fetcher.fetch("https://example.com/404")
            
            # 测试软404（内容为空/过短，触发ContentValidationError）
            # data = await fetcher.fetch("https://httpbin.org/status/200?content=")
        except (FetchError, ContentValidationError) as e:
            print(f"抓取/校验失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_news_fetcher())