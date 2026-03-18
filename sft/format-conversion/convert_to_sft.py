"""
SFT格式转换模块

将各种格式的数据转换为TRL训练格式：
- Messages格式（对话训练）
- Prompt-Completion格式
- Text格式
"""

import os
import json
import argparse
import re
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm


class SFTFormatConverter:
    """SFT格式转换器"""

    def __init__(self):
        pass

    def qa_to_messages(self, question: str, answer: str, context: str = "") -> Dict:
        """将QA格式转换为messages格式"""
        messages = []

        if context:
            messages.append({
                "role": "user",
                "content": f"背景信息：\n{context}\n\n问题：{question}"
            })
        else:
            messages.append({
                "role": "user",
                "content": question
            })

        messages.append({
            "role": "assistant",
            "content": answer
        })

        return {"messages": messages}

    def instruction_to_messages(self, instruction: str, input_text: str, output: str) -> Dict:
        """将Alpaca格式转换为messages格式"""
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        }

    def conversation_to_messages(self, conversation: List[Dict]) -> Dict:
        """将对话列表转换为messages格式"""
        return {"messages": conversation}

    def text_to_prompt_completion(self, prompt: str, completion: str) -> Dict:
        """转换为prompt-completion格式"""
        return {
            "prompt": prompt,
            "completion": completion
        }

    def code_to_messages(self, code: str, language: str, description: str = "") -> Dict:
        """将代码转换为messages格式"""
        user_content = f"请用{language}编写代码"
        if description:
            user_content = f"请用{language}编写代码：{description}"

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"```{language}\n{code}\n```"}
            ]
        }

    def summary_to_messages(self, original: str, summary: str) -> Dict:
        """将摘要对转换为messages格式"""
        return {
            "messages": [
                {"role": "user", "content": f"请总结以下内容：\n\n{original}"},
                {"role": "assistant", "content": summary}
            ]
        }

    def translation_to_messages(self, source_text: str, target_text: str,
                                 source_lang: str = "中文", target_lang: str = "英文") -> Dict:
        """将翻译对转换为messages格式"""
        return {
            "messages": [
                {"role": "user", "content": f"请将以下{source_lang}翻译成{target_lang}：\n\n{source_text}"},
                {"role": "assistant", "content": target_text}
            ]
        }


class BatchConverter:
    """批量转换器"""

    def __init__(self, converter: SFTFormatConverter):
        self.converter = converter

    def convert_file(
        self,
        input_path: str,
        output_path: str,
        input_format: str = "auto",
        output_format: str = "messages",
        min_length: int = 10,
        max_length: int = 50000
    ) -> Dict:
        """
        批量转换文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            input_format: 输入格式 (auto, qa, alpaca, conversation, sharegpt)
            output_format: 输出格式 (messages, prompt-completion)
            min_length: 最小文本长度
            max_length: 最大文本长度

        Returns:
            统计信息
        """
        stats = {
            "total": 0,
            "converted": 0,
            "skipped": 0,
            "errors": 0
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="转换中"):
                stats["total"] += 1

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    stats["errors"] += 1
                    continue

                converted = self._convert_sample(data, input_format, output_format)

                if converted is None:
                    stats["skipped"] += 1
                    continue

                # 检查长度
                text = self._get_text_length(converted)
                if text < min_length or text > max_length:
                    stats["skipped"] += 1
                    continue

                f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                stats["converted"] += 1

        return stats

    def _convert_sample(self, data: Dict, input_format: str, output_format: str) -> Optional[Dict]:
        """转换单个样本"""
        # 自动检测格式
        if input_format == "auto":
            input_format = self._detect_format(data)

        try:
            if input_format == "qa":
                if output_format == "messages":
                    return self.converter.qa_to_messages(
                        data.get("question", ""),
                        data.get("answer", ""),
                        data.get("context", "")
                    )
                else:
                    return self.converter.text_to_prompt_completion(
                        data.get("question", ""),
                        data.get("answer", "")
                    )

            elif input_format == "alpaca":
                return self.converter.instruction_to_messages(
                    data.get("instruction", ""),
                    data.get("input", ""),
                    data.get("output", "")
                )

            elif input_format == "conversation":
                return self.converter.conversation_to_messages(data.get("messages", []))

            elif input_format == "sharegpt":
                # ShareGPT格式转换
                conversations = data.get("conversations", [])
                messages = []
                for conv in conversations:
                    role = "user" if conv.get("from") == "human" else "assistant"
                    messages.append({"role": role, "content": conv.get("value", "")})
                return self.converter.conversation_to_messages(messages)

            elif input_format == "openassistant":
                # OpenAssistant格式
                messages = []
                for msg in data.get("messages", []):
                    messages.append({
                        "role": msg.get("role", ""),
                        "content": msg.get("content", "")
                    })
                return self.converter.conversation_to_messages(messages)

            elif input_format == "messages":
                # 已经是目标格式
                return data

            else:
                return None

        except Exception as e:
            return None

    def _detect_format(self, data: Dict) -> str:
        """自动检测数据格式"""
        if "messages" in data:
            return "messages"
        elif "question" in data and "answer" in data:
            return "qa"
        elif "instruction" in data:
            return "alpaca"
        elif "conversations" in data:
            return "sharegpt"
        else:
            return "unknown"

    def _get_text_length(self, data: Dict) -> int:
        """获取文本总长度"""
        if "messages" in data:
            return sum(len(m.get("content", "")) for m in data["messages"])
        elif "prompt" in data:
            return len(data.get("prompt", "")) + len(data.get("completion", ""))
        return 0


def main():
    parser = argparse.ArgumentParser(description="SFT格式转换")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--input-format", default="auto", help="输入格式")
    parser.add_argument("--output-format", default="messages", help="输出格式")
    parser.add_argument("--min-length", type=int, default=10, help="最小文本长度")
    parser.add_argument("--max-length", type=int, default=50000, help="最大文本长度")

    args = parser.parse_args()

    converter = SFTFormatConverter()
    batch_converter = BatchConverter(converter)

    stats = batch_converter.convert_file(
        args.input,
        args.output,
        args.input_format,
        args.output_format,
        args.min_length,
        args.max_length
    )

    print(f"转换完成:")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['converted']}")
    print(f"  跳过: {stats['skipped']}")
    print(f"  错误: {stats['errors']}")


if __name__ == "__main__":
    main()