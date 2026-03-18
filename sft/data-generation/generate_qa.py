"""
SFT数据生成模块 - API调用生成问答对

支持多种LLM API：
- DeepSeek
- OpenAI
- Claude
- 通义千问
- 智谱AI
"""

import os
import json
import time
import asyncio
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API配置"""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class QAPair:
    """问答对"""
    question: str
    answer: str
    context: str = ""
    metadata: Dict = field(default_factory=dict)


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.request_count = 0
        self.error_count = 0

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        pass

    @abstractmethod
    async def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成回复"""
        pass

    def _handle_error(self, error: Exception, retry_count: int) -> bool:
        """处理错误"""
        self.error_count += 1
        logger.error(f"API请求失败 (尝试 {retry_count}/{self.config.max_retries}): {str(error)}")
        return retry_count < self.config.max_retries


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url or "https://api.deepseek.com/v1"
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for retry in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model or "deepseek-chat",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                self.request_count += 1
                return response.choices[0].message.content
            except Exception as e:
                if not self._handle_error(e, retry):
                    raise
                await asyncio.sleep(self.config.retry_delay * (retry + 1))
        return ""

    async def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成"""
        tasks = [self.generate(p, system_prompt) for p in prompts]
        return await asyncio.gather(*tasks)


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url or "https://api.openai.com/v1"
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for retry in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model or "gpt-4o-mini",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                self.request_count += 1
                return response.choices[0].message.content
            except Exception as e:
                if not self._handle_error(e, retry):
                    raise
                await asyncio.sleep(self.config.retry_delay * (retry + 1))
        return ""

    async def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成"""
        tasks = [self.generate(p, system_prompt) for p in prompts]
        return await asyncio.gather(*tasks)


class ClaudeClient(BaseLLMClient):
    """Claude API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        for retry in range(self.config.max_retries):
            try:
                kwargs = {
                    "model": self.config.model or "claude-3-haiku-20240307",
                    "max_tokens": self.config.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = await self.client.messages.create(**kwargs)
                self.request_count += 1
                return response.content[0].text
            except Exception as e:
                if not self._handle_error(e, retry):
                    raise
                await asyncio.sleep(self.config.retry_delay * (retry + 1))
        return ""

    async def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成"""
        tasks = [self.generate(p, system_prompt) for p in prompts]
        return await asyncio.gather(*tasks)


class QwenClient(BaseLLMClient):
    """通义千问 API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            import dashscope
            dashscope.api_key = config.api_key
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        from dashscope import Generation

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for retry in range(self.config.max_retries):
            try:
                response = Generation.call(
                    model=self.config.model or "qwen-turbo",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    result_format='message'
                )
                self.request_count += 1
                return response.output.choices[0].message.content
            except Exception as e:
                if not self._handle_error(e, retry):
                    raise
                await asyncio.sleep(self.config.retry_delay * (retry + 1))
        return ""

    async def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成"""
        results = []
        for p in prompts:
            result = await self.generate(p, system_prompt)
            results.append(result)
        return results


class ZhipuClient(BaseLLMClient):
    """智谱AI API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("请安装 zhipuai: pip install zhipuai")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for retry in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model or "glm-4-flash",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                self.request_count += 1
                return response.choices[0].message.content
            except Exception as e:
                if not self._handle_error(e, retry):
                    raise
                await asyncio.sleep(self.config.retry_delay * (retry + 1))
        return ""

    async def generate_batch(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成"""
        results = []
        for p in prompts:
            result = await self.generate(p, system_prompt)
            results.append(result)
        return results


def get_client(api_type: str, config: APIConfig) -> BaseLLMClient:
    """获取API客户端"""
    clients = {
        "deepseek": DeepSeekClient,
        "openai": OpenAIClient,
        "claude": ClaudeClient,
        "qwen": QwenClient,
        "zhipu": ZhipuClient,
    }

    if api_type.lower() not in clients:
        raise ValueError(f"不支持的API类型: {api_type}，支持: {list(clients.keys())}")

    return clients[api_type.lower()](config)


class QAGenerator:
    """问答对生成器"""

    # 不同领域的系统提示词模板
    SYSTEM_PROMPTS = {
        "general": "你是一个专业的AI助手，请根据给定的上下文生成高质量的问答对。",
        "technical": "你是一个技术专家，请根据给定的技术文档生成专业的问答对，确保技术准确性。",
        "academic": "你是一个学术研究者，请根据给定的学术内容生成深入的问答对。",
        "code": "你是一个编程专家，请根据给定的代码生成关于代码功能、用法和最佳实践的问答对。",
        "energy": "你是能源领域的专家，请根据给定的能源相关内容生成专业的问答对。",
    }

    def __init__(self, client: BaseLLMClient, domain: str = "general"):
        self.client = client
        self.domain = domain

    def _build_qa_prompt(self, context: str, num_questions: int = 3, style: str = "detailed") -> str:
        """构建生成问答对的提示词"""
        prompt = f"""请根据以下内容生成{num_questions}个高质量的问答对。

要求：
1. 问题应该清晰、具体，涵盖内容的不同方面
2. 答案应该详细、准确，直接基于给定的内容
3. 问答对应该有助于理解和应用这些知识
4. 格式要求：每对问答使用Q:和A:标记

内容：
{context}

请生成{num_questions}个问答对："""
        return prompt

    def _parse_qa_pairs(self, response: str, context: str = "") -> List[QAPair]:
        """解析API返回的问答对"""
        qa_pairs = []

        # 尝试解析问答对
        lines = response.strip().split('\n')
        current_q = ""
        current_a = ""

        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('问题:') or line.startswith('Q：'):
                if current_q and current_a:
                    qa_pairs.append(QAPair(
                        question=current_q,
                        answer=current_a,
                        context=context
                    ))
                current_q = line[2:].strip() if line[1] in [':', '：'] else line[3:].strip()
                current_a = ""
            elif line.startswith('A:') or line.startswith('答案:') or line.startswith('A：'):
                current_a = line[2:].strip() if line[1] in [':', '：'] else line[3:].strip()
            elif current_q and not current_a:
                # 答案可能跨多行
                pass
            elif current_q and current_a:
                current_a += "\n" + line

        # 添加最后一对
        if current_q and current_a:
            qa_pairs.append(QAPair(
                question=current_q,
                answer=current_a,
                context=context
            ))

        return qa_pairs

    async def generate_qa_pairs(
        self,
        context: str,
        num_questions: int = 3,
        style: str = "detailed"
    ) -> List[QAPair]:
        """生成问答对"""
        system_prompt = self.SYSTEM_PROMPTS.get(self.domain, self.SYSTEM_PROMPTS["general"])
        prompt = self._build_qa_prompt(context, num_questions, style)

        response = await self.client.generate(prompt, system_prompt)
        return self._parse_qa_pairs(response, context)

    async def generate_qa_pairs_batch(
        self,
        contexts: List[str],
        num_questions: int = 3,
        concurrency: int = 5
    ) -> List[QAPair]:
        """批量生成问答对"""
        all_qa_pairs = []
        semaphore = asyncio.Semaphore(concurrency)

        async def process_context(context: str) -> List[QAPair]:
            async with semaphore:
                try:
                    return await self.generate_qa_pairs(context, num_questions)
                except Exception as e:
                    logger.error(f"处理上下文失败: {str(e)}")
                    return []

        tasks = [process_context(ctx) for ctx in contexts]
        results = await asyncio.gather(*tasks)

        for qa_pairs in results:
            all_qa_pairs.extend(qa_pairs)

        return all_qa_pairs


class SFTDataGenerator:
    """SFT数据生成器"""

    def __init__(self, client: BaseLLMClient, output_path: str):
        self.client = client
        self.qa_generator = QAGenerator(client)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    async def generate_from_texts(
        self,
        texts: List[str],
        num_questions_per_text: int = 3,
        concurrency: int = 5
    ) -> int:
        """从文本列表生成问答对"""
        qa_pairs = await self.qa_generator.generate_qa_pairs_batch(
            texts,
            num_questions_per_text,
            concurrency
        )

        # 写入文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": qa.question},
                        {"role": "assistant", "content": qa.answer}
                    ],
                    "context": qa.context
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"生成 {len(qa_pairs)} 个问答对，保存到 {self.output_path}")
        return len(qa_pairs)

    async def generate_from_file(
        self,
        input_path: str,
        text_field: str = "text",
        num_questions_per_text: int = 3,
        concurrency: int = 5
    ) -> int:
        """从文件生成问答对"""
        texts = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    text = data.get(text_field, "")
                    if text:
                        texts.append(text)

        return await self.generate_from_texts(texts, num_questions_per_text, concurrency)


async def main():
    parser = argparse.ArgumentParser(description="使用LLM API生成SFT问答对")
    parser.add_argument("--api", required=True, choices=["deepseek", "openai", "claude", "qwen", "zhipu"],
                        help="API类型")
    parser.add_argument("--api-key", help="API密钥（也可通过环境变量设置）")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--num-questions", type=int, default=3, help="每段文本生成的问题数")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数")
    parser.add_argument("--domain", default="general",
                        choices=["general", "technical", "academic", "code", "energy"],
                        help="领域类型")

    args = parser.parse_args()

    # 获取API密钥
    api_key = args.api_key or os.getenv(f"{args.api.upper()}_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(f"请提供API密钥: --api-key 或设置环境变量 {args.api.upper()}_API_KEY")

    # 创建配置
    config = APIConfig(
        api_key=api_key,
        model=args.model
    )

    # 创建客户端和生成器
    client = get_client(args.api, config)
    qa_generator = QAGenerator(client, args.domain)
    generator = SFTDataGenerator(client, args.output)
    generator.qa_generator = qa_generator

    # 生成数据
    count = await generator.generate_from_file(
        args.input,
        num_questions_per_text=args.num_questions,
        concurrency=args.concurrency
    )

    print(f"完成！共生成 {count} 个问答对")


if __name__ == "__main__":
    asyncio.run(main())