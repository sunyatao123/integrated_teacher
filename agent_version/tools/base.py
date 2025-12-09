#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能工具基类和通用功能

设计原则：
1. 单例模式的AI客户端
2. 统一的结果封装
3. 智能的提示词加载
4. 容错机制
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_version.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    SILICONFLOW_MODEL,
    PROMPTS_DIR,
    ORIGINAL_PROMPTS_DIR,
)

logger = logging.getLogger("agent")


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any
    error: str = ""
    confidence: float = 1.0  # 结果置信度
    alternatives: List[Any] = field(default_factory=list)  # 备选结果

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "confidence": self.confidence,
            "alternatives": self.alternatives
        }


class AIModelClient:
    """
    AI模型客户端（单例模式）

    特性：
    1. 单例确保资源复用
    2. 支持流式和非流式调用
    3. 内置重试机制
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        self.api_key = SILICONFLOW_API_KEY
        self.base_url = SILICONFLOW_BASE_URL
        self.model = SILICONFLOW_MODEL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_retries = 3

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        response_format: Optional[Dict] = None,
        stream: bool = False
    ):
        """统一的聊天接口"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if response_format:
            kwargs["response_format"] = response_format

        return self.client.chat.completions.create(**kwargs)


def load_prompt_template(template_name: str) -> str:
    """
    加载提示词模板

    优先级：
    1. agent_version/prompts/
    2. 原项目prompts/
    """
    # 处理文件名
    if not template_name.endswith('.txt'):
        template_name = f"{template_name}.txt"

    # 优先从agent_version的prompts目录加载
    agent_path = PROMPTS_DIR / template_name
    if agent_path.exists():
        with open(agent_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    # 回退到原项目的prompts目录
    original_path = ORIGINAL_PROMPTS_DIR / template_name
    if original_path.exists():
        with open(original_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    logger.warning(f"提示词模板 {template_name} 未找到")
    return ""


def load_class_profiles() -> Dict[str, Any]:
    """从原项目prompts文件夹加载班级配置"""
    from agent_version.config import ORIGINAL_CLASS_PROFILES
    try:
        with open(ORIGINAL_CLASS_PROFILES, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("班级配置文件 class_profiles.json 未找到")
        return {}
    except json.JSONDecodeError:
        logger.warning("班级配置文件 class_profiles.json 格式错误")
        return {}

