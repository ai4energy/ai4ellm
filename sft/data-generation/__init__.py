"""
SFT数据生成模块初始化
"""

from .generate_qa import (
    get_client,
    QAGenerator,
    SFTDataGenerator,
    APIConfig,
    DeepSeekClient,
    OpenAIClient,
    ClaudeClient,
    QwenClient,
    ZhipuClient,
)

from .generate_multi_type import (
    ConversationGenerator,
    SummaryGenerator,
    TranslationGenerator,
    CodeExplanationGenerator,
    InstructionGenerator,
)

__all__ = [
    # API客户端
    'get_client',
    'APIConfig',
    'DeepSeekClient',
    'OpenAIClient',
    'ClaudeClient',
    'QwenClient',
    'ZhipuClient',

    # 生成器
    'QAGenerator',
    'SFTDataGenerator',
    'ConversationGenerator',
    'SummaryGenerator',
    'TranslationGenerator',
    'CodeExplanationGenerator',
    'InstructionGenerator',
]