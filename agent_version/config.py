#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent版本配置文件
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent
ORIGINAL_PROJECT_DIR = BASE_DIR.parent

# API配置
SILICONFLOW_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    "sk-gsaohhwdtofontqfoaqistheumksejbrhypkpfrolhikfqmy"
)
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1"
)
SILICONFLOW_MODEL = os.getenv(
    "SILICONFLOW_MODEL",
    "deepseek-ai/DeepSeek-V3"
)

# 检索服务配置
SEARCH_BASE_URL = os.getenv("SEARCH_BASE_URL", "http://127.0.0.1:8001")

# 日志配置
DEBUG_AI = os.getenv('DEBUG_AI', '1') == '1'

# 文件路径配置
PROMPTS_DIR = BASE_DIR / "prompts"
TEMPLATES_DIR = BASE_DIR / "templates"
LOGS_DIR = BASE_DIR / "logs"

# 原项目文件路径（用于复用）
ORIGINAL_PROMPTS_DIR = ORIGINAL_PROJECT_DIR / "prompts"
ORIGINAL_CLASS_PROFILES = ORIGINAL_PROMPTS_DIR / "class_profiles.json"

# Agent配置
AGENT_MAX_ITERATIONS = 10
AGENT_VERBOSE = DEBUG_AI

# 工具配置
TOOL_TIMEOUT = 15.0
SEARCH_TIMEOUT = 8.0

