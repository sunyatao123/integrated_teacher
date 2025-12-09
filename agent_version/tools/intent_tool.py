#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别工具
"""

import json
import logging
from typing import Optional, Type, List, Dict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from .base import AIModelClient, load_prompt_template, ToolResult

logger = logging.getLogger("agent")


class IntentRecognitionInput(BaseModel):
    """意图识别工具输入"""
    user_text: str = Field(description="用户当前输入的文本")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="对话历史记录，格式为[{'role': 'user/assistant', 'content': '...'}]"
    )


class IntentRecognitionTool(BaseTool):
    """
    意图识别工具
    
    判断用户是想进行：
    - sports_meeting: 全员运动会方案设计
    - lesson_plan: 课课练方案设计  
    - chat: 闲聊或其他
    """
    name: str = "intent_recognition"
    description: str = """识别用户的意图类型。
用于判断用户是想要：
- sports_meeting（全员运动会方案设计）
- lesson_plan（课课练方案设计）
- chat（闲聊或其他问题）

输入用户文本和对话历史，返回意图类型。"""
    args_schema: Type[BaseModel] = IntentRecognitionInput
    
    def _run(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """执行意图识别"""
        try:
            result = self._detect_intent(user_text, conversation_history)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return json.dumps({"success": False, "data": "chat", "error": str(e)})
    
    def _detect_intent(
        self,
        user_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ToolResult:
        """使用大模型进行意图识别"""
        client = AIModelClient()
        system = load_prompt_template("intent_recognition")
        
        # 构建历史对话上下文
        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            lines = []
            for msg in recent:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"用户：{content}")
                elif role == "assistant":
                    lines.append(f"助手：{content}")
            history_text = "\n".join(lines) if lines else ""
        
        user_prompt = f"""
对话历史（最近6轮）：
{history_text if history_text else "（无历史记录）"}

当前用户输入：{user_text}

请判断用户的意图，只输出JSON。
""".strip()
        
        resp = client.client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        
        # 解析JSON
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(content[start:end+1])
                intent = parsed.get("intent", "chat")
                if intent in ("sports_meeting", "lesson_plan", "chat"):
                    return ToolResult(success=True, data=intent)
            except Exception:
                pass
        
        return ToolResult(success=True, data="chat")

