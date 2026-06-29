"""
清洗规则常量模块

集中定义文本清洗过程中使用的正则表达式、关键词集合等。
所有规则均可通过 config.yaml 覆盖。
"""

import re

# 默认跳过的章节名称（小写匹配）
DEFAULT_SKIP_SECTIONS: set[str] = {
    "前言", "思考题", "目录", "习题", "参考文献", "图书在版编目",
    "目 录", "目 次", "acknowledgements", "conference papers",
    "table of contents", "list of figures", "list of tables",
    "例题", "magnetism",
}

# 水印关键词（小写匹配）
DEFAULT_WATERMARK_KEYWORDS: list[str] = [
    "copyright", "email", "www.", "西安交通大学", "xianjiaotonguniversity",
]

# 行内引用匹配模式
INLINE_CITATION_RE = re.compile(r"\[.*?]")

# 作者-年份引用匹配（注意：必须用完整 Unicode 范围 一-龥）
AUTHOR_YEAR_RE = re.compile(
    r"\b[\w一-龥]+(?:和[\w一-龥]+)?\s*[（(]\d{4}(?:[，,]\d{4})*[）)]"
)

# 乱码字符匹配（保留 ASCII 可打印 + CJK + 全角标点）
GARBLE_RE = re.compile(r"[^\x20-\x7E一-龥　-〿＀-￯]")

# 页码行尾匹配
PAGE_NUM_TRAILING_RE = re.compile(r'[\/\s]*\(\s*\d{1,4}\s*\)\s*$')
PAGE_NUM_TRAILING2_RE = re.compile(r'[\/\s]+\d{1,4}\s*$')

# 图/表编号
TABLE_RE = re.compile(r"表\s?\d+[\-.\d]*")
FIGURE_RE = re.compile(r"图\s?\d+[\-.\d]*")
TABLE_EN_RE = re.compile(r"Table\s?\d+[\-.\d]*", re.IGNORECASE)
FIGURE_EN_RE = re.compile(r"Figure\s?\d+[\-.\d]*", re.IGNORECASE)

# 交叉引用
CROSS_REF_CN = re.compile(
    r'(?:如|见|参见)?\s*(?:图|表)\s*\d+(?:[.\-]\d+)*(?:[A-Za-z])?\s*(?:所示|所列|所给)?'
)
CROSS_REF_EN = re.compile(
    r'(?:see|as\s+shown\s+in)?\s*(?:figure|fig\.?|table)\s+\d+(?:[.\-]\d+)*(?:[A-Za-z])?',
    re.IGNORECASE,
)

# 章节目录行匹配
TOC_CHAPTER_RE = re.compile(
    r'^#?\s*第(?:[一二三四五六七八九十百]|\d+)章[^\n]*?[\/\s]+\d{1,4}\s*$'
)
TOC_SECTION_RE = re.compile(
    r'^\d+(?:\.\d+)+[^\n]*?(?:\(\d{1,4}\)|[\/\s]+\d{1,4})\s*$'
)

# 参考文献标记 (Author et al., 2024)
REF_PAREN_RE = re.compile(r"\([A-Za-z一-龥]+\s+et\s+al\.,\s+\d{4}\)")

# 中英文 URL 整行匹配
URL_LINE_RE = re.compile(r'^https?://\S+$')

# 西安交通大学水印
WATERMARK_XJTU = re.compile(r'西安\s*交通\s*大学\s*XIANJIAOTONGUNIVERSITY', re.IGNORECASE)

# 中英文统计字符数
CN_WORD_RE = re.compile(r"[\w一-龥]")
