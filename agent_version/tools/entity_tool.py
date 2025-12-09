#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体提取工具
"""

import json
import logging
import re
from typing import Optional, Type, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from .base import AIModelClient, load_prompt_template, load_class_profiles, ToolResult

logger = logging.getLogger("agent")


class EntityExtractionInput(BaseModel):
    """实体提取工具输入"""
    user_text: str = Field(description="用户当前输入的文本")
    plan_type: str = Field(description="意图类型: sports_meeting/lesson_plan/chat")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="对话历史记录"
    )


class EntityExtractionTool(BaseTool):
    """
    实体提取工具
    
    从用户输入中提取关键参数：
    - 课课练：年级(grades_query)、薄弱项(trained_weaknesses)
    - 运动会：操场条件(semantic_query)、人数(count_query)、年级
    """
    name: str = "entity_extraction"
    description: str = """从用户输入中提取关键参数信息。

课课练场景需要提取：
- grades_query: 年级（如"1"、"3"、"5"）
- trained_weaknesses: 薄弱项（如"速度"、"力量"、"柔韧"）

运动会场景需要提取：
- semantic_query: 操场条件描述
- count_query: 学生人数
- grades_query: 年级

返回提取到的参数和缺失的字段列表。"""
    args_schema: Type[BaseModel] = EntityExtractionInput
    
    def _run(
        self,
        user_text: str,
        plan_type: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """执行实体提取"""
        try:
            result = self._extract_entities(user_text, plan_type, conversation_history)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return json.dumps({"success": False, "data": {}, "error": str(e)})
    
    def _detect_class_and_fill_params(self, user_text: str) -> tuple:
        """检测班级名称并自动填充参数"""
        class_profiles = load_class_profiles()
        if not class_profiles:
            return False, {}
        
        # 中文数字转换
        cn_num_map = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
        }
        normalized_text = user_text
        for cn, num in cn_num_map.items():
            normalized_text = normalized_text.replace(cn, num)
        
        # 按班级名称长度排序匹配
        sorted_classes = sorted(class_profiles.items(), key=lambda x: len(x[0]), reverse=True)
        
        for class_name, class_info in sorted_classes:
            normalized_class = class_name
            for cn, num in cn_num_map.items():
                normalized_class = normalized_class.replace(cn, num)
            
            if normalized_class in normalized_text:
                params = {
                    "semantic_query": class_info.get("semantic_query", ""),
                    "count_query": class_info.get("count_query", ""),
                    "grades_query": class_info.get("grades_query", ""),
                    "trained_weaknesses": class_info.get("trained_weaknesses", ""),
                    "detected_class_name": class_name
                }
                return True, params
        
        return False, {}
    
    def _extract_entities(
        self,
        user_text: str,
        plan_type: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ToolResult:
        """使用大模型进行实体提取"""
        # 首先检测班级
        is_class, class_params = self._detect_class_and_fill_params(user_text)
        accumulated = dict(class_params) if is_class else {}
        
        client = AIModelClient()
        system = load_prompt_template("param_extraction_system")
        
        # 构建历史上下文
        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]
            lines = []
            for msg in recent:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"用户：{content}")
                elif role == "assistant":
                    lines.append(f"助手：{content}")
            history_text = "\n".join(lines) if lines else ""
        
        user_template = load_prompt_template("param_extraction_user")
        user_prompt = user_template.format(
            history_text=history_text if history_text else "（无历史记录）",
            user_text=user_text,
        )
        
        resp = client.client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.2,
            response_format={'type': 'json_object'},
        )
        content = resp.choices[0].message.content.strip()
        
        # 解析JSON
        start = content.find("{")
        end = content.rfind("}")
        parsed = {}
        if start != -1 and end != -1:
            try:
                parsed = json.loads(content[start:end+1])
            except Exception:
                pass
        
        # 合并参数
        for key, value in parsed.items():
            if value is not None and value != "":
                accumulated[key] = value
        
        # 判断缺失字段
        missing = []
        if plan_type == "sports_meeting":
            if not accumulated.get("semantic_query"):
                missing.append("semantic_query")
        elif plan_type == "lesson_plan":
            if not accumulated.get("grades_query") and not accumulated.get("trained_weaknesses"):
                missing.extend(["grades_query", "trained_weaknesses"])
            elif not accumulated.get("grades_query"):
                missing.append("grades_query")
            elif not accumulated.get("trained_weaknesses"):
                missing.append("trained_weaknesses")
        
        return ToolResult(
            success=True,
            data={"params": accumulated, "missing": missing, "plan_type": plan_type}
        )

