from openai import OpenAI
import config
from typing import List, Dict

class OpenAIClient:
    """自定义 OpenAI API 客户端，支持自定义 URL"""
    
    def __init__(self, api_key: str = None, base_url: str = None, timeout: float = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        if not self.api_key or not self.base_url:
            raise ValueError("Explicit API key and base URL are required")
        self.timeout = timeout if timeout is not None else config.OPENAI_TIMEOUT_SECONDS
        
        # 创建 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
    
    def chat_completion(self, 
                      messages: List[Dict[str, str]], 
                      model: str = None,
                      temperature: float = 0.7,
                      max_tokens: int = config.MAX_TOKENS,
                      stream: bool = False) -> str:
        """调用 ChatCompletion API 获取回复"""
        model = model or config.DEFAULT_MODEL
        if not model:
            raise ValueError("Explicit model is required")
        try:
            if not stream:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                # 流式响应处理
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                # 收集流式响应内容
                collected_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        collected_content += content_chunk
                        # 可以在这里添加实时处理逻辑
                        
                return collected_content
                
        except Exception as e:
            print(f"OpenAI API 调用失败: {e}")
            if not config.FALLBACK_MODEL:
                raise
            # 仅使用显式配置的备用模型
            try:
                response = self.client.chat.completions.create(
                    model=config.FALLBACK_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"备用模型调用也失败: {e2}")
                # 不再返回错误消息，而是抛出异常
                raise Exception(f"API调用失败，主模型错误：{e}，备用模型错误：{e2}")
