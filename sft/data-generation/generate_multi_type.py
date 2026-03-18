"""
SFT数据生成 - 多类型数据生成

支持生成多种类型的训练数据：
- 问答对
- 摘要
- 翻译
- 代码解释
- 多轮对话
"""

import os
import json
import asyncio
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """对话轮次"""
    role: str
    content: str


@dataclass
class MultiTurnConversation:
    """多轮对话"""
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_messages(self) -> List[Dict]:
        """转换为messages格式"""
        return [{"role": t.role, "content": t.content} for t in self.turns]


class ConversationGenerator:
    """多轮对话生成器"""

    CONVERSATION_TEMPLATES = {
        "qa": {
            "system": "你是一个知识渊博的助手，能够准确回答用户的问题。",
            "start": "请问有什么可以帮助您的？",
        },
        "code": {
            "system": "你是一个编程专家，能够帮助用户解决编程问题并提供代码示例。",
            "start": "我可以帮助您解决编程问题，请问您需要什么帮助？",
        },
        "analysis": {
            "system": "你是一个分析专家，能够深入分析问题并提供见解。",
            "start": "我可以帮您分析问题，请告诉我您想了解什么？",
        },
        "energy": {
            "system": "你是能源领域的专家，能够回答关于能源系统、能源政策、能源技术等问题。",
            "start": "我是能源领域的专家，请问有什么能源相关的问题需要咨询？",
        },
    }

    def __init__(self, client, conversation_type: str = "qa"):
        self.client = client
        self.conversation_type = conversation_type

    async def generate_multi_turn(
        self,
        topic: str,
        num_turns: int = 3,
        context: str = ""
    ) -> MultiTurnConversation:
        """生成多轮对话"""
        template = self.CONVERSATION_TEMPLATES.get(self.conversation_type, self.CONVERSATION_TEMPLATES["qa"])

        prompt = f"""请生成一个关于"{topic}"的{num_turns}轮对话。

要求：
1. 对话自然流畅，符合真实用户场景
2. 每轮对话要有实质性内容
3. 用户问题应该有递进性，逐步深入
4. 助手回答应该详细、有帮助

{"上下文：" + context if context else ""}

请按以下格式输出：
User: 用户问题1
Assistant: 助手回答1
User: 用户问题2
Assistant: 助手回答2
..."""

        response = await self.client.generate(prompt, template["system"])
        return self._parse_conversation(response)

    def _parse_conversation(self, response: str) -> MultiTurnConversation:
        """解析对话"""
        conversation = MultiTurnConversation()
        lines = response.strip().split('\n')

        current_role = ""
        current_content = []

        for line in lines:
            line = line.strip()
            if line.startswith("User:") or line.startswith("用户:"):
                if current_role and current_content:
                    conversation.turns.append(ConversationTurn(
                        role=current_role,
                        content="\n".join(current_content).strip()
                    ))
                current_role = "user"
                current_content = [line.split(":", 1)[1].strip()]
            elif line.startswith("Assistant:") or line.startswith("助手:"):
                if current_role and current_content:
                    conversation.turns.append(ConversationTurn(
                        role=current_role,
                        content="\n".join(current_content).strip()
                    ))
                current_role = "assistant"
                current_content = [line.split(":", 1)[1].strip()]
            elif current_role:
                current_content.append(line)

        if current_role and current_content:
            conversation.turns.append(ConversationTurn(
                role=current_role,
                content="\n".join(current_content).strip()
            ))

        return conversation


class SummaryGenerator:
    """摘要生成器"""

    async def generate_summary(self, client, text: str, style: str = "brief") -> str:
        """生成摘要"""
        style_prompts = {
            "brief": "请用1-2句话概括以下内容的核心要点：",
            "detailed": "请详细概括以下内容，包括主要观点和关键细节：",
            "bullet": "请用要点列表的形式概括以下内容：",
        }

        prompt = f"{style_prompts.get(style, style_prompts['brief'])}\n\n{text}"
        return await client.generate(prompt, "你是一个专业的文本摘要专家。")


class TranslationGenerator:
    """翻译数据生成器"""

    LANGUAGE_PAIRS = {
        ("zh", "en"): ("中文", "英文"),
        ("en", "zh"): ("英文", "中文"),
        ("zh", "ja"): ("中文", "日语"),
        ("ja", "zh"): ("日语", "中文"),
    }

    async def generate_translation(
        self,
        client,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en"
    ) -> Dict[str, str]:
        """生成翻译数据"""
        lang_names = self.LANGUAGE_PAIRS.get((source_lang, target_lang), (source_lang, target_lang))

        prompt = f"""请将以下{lang_names[0]}文本翻译成{lang_names[1]}，要求翻译准确、自然流畅：

{text}"""

        translation = await client.generate(prompt, "你是一个专业的翻译专家。")

        return {
            "source_text": text,
            "source_lang": source_lang,
            "target_text": translation,
            "target_lang": target_lang
        }


class CodeExplanationGenerator:
    """代码解释生成器"""

    async def generate_explanation(self, client, code: str, language: str = "python") -> Dict[str, str]:
        """生成代码解释"""
        prompt = f"""请解释以下{language}代码的功能和实现逻辑：

```{language}
{code}
```

请包含：
1. 代码功能概述
2. 关键步骤说明
3. 可能的改进建议"""

        explanation = await client.generate(
            prompt,
            "你是一个编程专家，能够清晰地解释代码的功能和原理。"
        )

        return {
            "code": code,
            "language": language,
            "explanation": explanation
        }


class InstructionGenerator:
    """指令数据生成器"""

    INSTRUCTION_TEMPLATES = {
        "classification": """根据以下文本，判断其所属类别。

类别选项：{categories}

文本：{text}

请直接输出类别名称。""",

        "extraction": """从以下文本中提取所有{entity_type}。

文本：{text}

请以列表形式输出提取结果。""",

        "generation": """请根据以下要求生成内容。

要求：{instruction}
{"约束条件：" + constraints if constraints else ""}

请生成符合要求的内容。""",

        "reasoning": """请分析以下问题并给出推理过程。

问题：{question}

{"已知条件：" + context if context else ""}

请逐步分析并给出答案。"""
    }

    async def generate_instruction_data(
        self,
        client,
        instruction_type: str,
        **kwargs
    ) -> Dict[str, str]:
        """生成指令数据"""
        template = self.INSTRUCTION_TEMPLATES.get(instruction_type, "")
        if not template:
            raise ValueError(f"不支持的指令类型: {instruction_type}")

        prompt = template.format(**kwargs)
        response = await client.generate(prompt, "你是一个能够准确理解和执行指令的AI助手。")

        return {
            "instruction": prompt,
            "output": response,
            "type": instruction_type
        }


async def main():
    """主函数示例"""
    parser = argparse.ArgumentParser(description="SFT数据生成")
    parser.add_argument("--type", choices=["conversation", "summary", "translation", "code", "instruction"],
                        required=True, help="生成类型")
    parser.add_argument("--input", "-i", help="输入文件")
    parser.add_argument("--output", "-o", required=True, help="输出文件")
    parser.add_argument("--api", default="deepseek", help="API类型")
    parser.add_argument("--api-key", help="API密钥")

    args = parser.parse_args()

    # 这里需要根据实际情况初始化client
    print(f"生成 {args.type} 类型数据到 {args.output}")


if __name__ == "__main__":
    asyncio.run(main())