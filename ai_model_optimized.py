#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模型调用模块 - 教师端专用版本

用于教师端备课助手和班级数据分析
提供统一的API调用接口
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from openai import OpenAI


# 配置日志
def setup_ai_logger():
    """配置AI模型日志系统"""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("ai_model")
    logger.setLevel(logging.DEBUG if os.getenv('DEBUG_AI', '1') == '1' else logging.INFO)

    if logger.handlers:
        return logger

    log_file = log_dir / "ai_model.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_ai_logger()


class OptimizedAIModel:
    """
    AI模型封装类 - 教师端专用
    
    提供统一的API调用接口，用于：
    1. 教师端备课助手
       - 意图识别（detect_intent_llm）
       - 参数提取（collect_entities_llm）
       - 方案生成（generate_plan, generate_plan_stream）
       - 闲聊回复
    2. 班级数据分析
       - 薄弱项识别（analyze_with_llm）
    
    使用方式：
        model = OptimizedAIModel()
        response = model.client.chat.completions.create(
            model=model.model,
            messages=[...],
            ...
        )
    """
    
    def __init__(self):
        """
        初始化AI模型配置
        
        从环境变量读取配置：
        - SILICONFLOW_API_KEY: API密钥
        - SILICONFLOW_BASE_URL: API基础URL
        - SILICONFLOW_MODEL: 模型名称
        """
        self.api_key = os.getenv(
            "SILICONFLOW_API_KEY",
            "sk-gsaohhwdtofontqfoaqistheumksejbrhypkpfrolhikfqmy"
        )
        self.base_url = os.getenv(
            "SILICONFLOW_BASE_URL",
            "https://api.siliconflow.cn/v1"
        )
        self.model = os.getenv(
            "SILICONFLOW_MODEL",
            "deepseek-ai/DeepSeek-V3"
        )
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        if os.getenv('DEBUG_AI', '1') == '1':
            logger.debug(f"✅ AI模型初始化完成")
            logger.debug(f"   模型: {self.model}")
            logger.debug(f"   Base URL: {self.base_url}")
            logger.debug(f"   API Key: {self.api_key[:20]}...{self.api_key[-10:]}")
