#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
思考引擎 - Agent的深度思考能力

实现：
1. 意图理解：理解显性、隐性、潜在需求
2. 上下文构建：关联历史、环境、业务目标
3. 目标分解：将模糊需求转化为具体子目标
4. 约束识别：识别显性和隐性约束条件
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from agent_version.agent.memory import UserGoal
from agent_version.tools.base import AIModelClient, load_prompt_template

logger = logging.getLogger("agent_thinking")


@dataclass
class ThoughtResult:
    """思考结果"""
    understood_intent: str  # 理解的意图
    explicit_needs: List[str]  # 显性需求
    implicit_needs: List[str]  # 隐性需求
    potential_needs: List[str]  # 潜在需求
    constraints: List[str]  # 约束条件
    sub_goals: List[str]  # 分解的子目标
    confidence: float  # 理解置信度
    reasoning: str  # 推理过程
    suggested_approach: str  # 建议的处理方式
    needs_clarification: bool  # 是否需要澄清
    clarification_questions: List[str] = field(default_factory=list)  # 澄清问题


class ThinkingEngine:
    """
    思考引擎
    
    核心能力：
    1. 深度理解用户意图
    2. 挖掘隐性和潜在需求
    3. 智能分解目标
    4. 识别约束条件
    """
    
    def __init__(self):
        self.client = AIModelClient()
    
    def analyze_request(
        self,
        user_input: str,
        context: Dict[str, Any]
    ) -> ThoughtResult:
        """
        深度分析用户请求
        
        参数:
            user_input: 用户输入
            context: 完整上下文（来自记忆系统）
            
        返回:
            ThoughtResult: 思考结果
        """
        # 对于简单闲聊/功能咨询等请求，优先走轻量级快速路径，减少一次大模型调用
        if self._should_use_fast_path(user_input, context):
            logger.info("[Thinking] 使用快速路径（规则+降级），跳过LLM深度分析")
            return self._create_fallback_result(user_input)

        # 构建思考提示词
        thinking_prompt = self._build_thinking_prompt(user_input, context)
        
        try:
            response = self.client.client.chat.completions.create(
                model=self.client.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": thinking_prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_thought_result(content)
            
        except Exception as e:
            logger.error(f"思考引擎分析失败: {e}")
            return self._create_fallback_result(user_input)
    
    def _get_system_prompt(self) -> str:
        """获取思考引擎系统提示词"""
        return """你是一个智能思考引擎，负责深度理解用户的真实意图和需求。

你的任务是：
1. **理解显性需求**：用户明确表达的内容
2. **挖掘隐性需求**：用户没有明说但实际需要的
3. **预测潜在需求**：用户可能还没意识到的需求
4. **识别约束条件**：明确的和隐含的限制
5. **分解子目标**：将大目标拆分为可执行的小目标
6. **评估理解置信度**：对自己理解的信心程度

你需要像一个经验丰富的助手一样思考：
- 用户为什么会提出这个请求？
- 用户的最终目标是什么？
- 有什么是用户没说但很重要的？
- 怎样才能最好地满足用户？

输出JSON格式的分析结果。"""
    
    def _build_thinking_prompt(
        self, 
        user_input: str, 
        context: Dict[str, Any]
    ) -> str:
        """构建思考提示词"""
        # 提取关键上下文
        history = context.get("conversation_history", [])
        collected_params = context.get("collected_params", {})
        user_goals = context.get("user_goals", [])
        accumulated = context.get("accumulated_params", {})
        
        # 格式化历史对话
        history_text = ""
        if history:
            recent = history[-6:]
            for msg in recent:
                role = "用户" if msg.get("role") == "user" else "助手"
                history_text += f"{role}: {msg.get('content', '')[:200]}...\n"
        
        # 格式化已收集的参数
        params_text = json.dumps(collected_params, ensure_ascii=False, indent=2) if collected_params else "无"
        
        # 格式化累积参数
        accumulated_text = json.dumps(accumulated, ensure_ascii=False, indent=2)
        
        return f"""请深度分析以下用户请求：

## 当前用户输入
{user_input}

## 对话历史（最近6轮）
{history_text if history_text else "无历史记录"}

## 已收集的参数
{params_text}

## 累积的业务参数
{accumulated_text}

## 分析要求
请输出JSON格式的分析结果，包含以下字段：
{{
    "understood_intent": "理解的核心意图（sports_meeting/lesson_plan/chat/clarification）",
    "explicit_needs": ["显性需求列表"],
    "implicit_needs": ["隐性需求列表（用户没说但需要的）"],
    "potential_needs": ["潜在需求列表（可以进一步提供的）"],
    "constraints": ["约束条件列表"],
    "sub_goals": ["分解的子目标列表"],
    "confidence": 0.0-1.0之间的置信度,
    "reasoning": "推理过程说明",
    "suggested_approach": "建议的处理方式",
    "needs_clarification": true/false,
    "clarification_questions": ["如果需要澄清，列出问题"]
	}}"""

    def _should_use_fast_path(self, user_input: str, context: Dict[str, Any]) -> bool:
        """判断是否可以使用快速思考路径（不调用LLM）

        目标：
        - 对「你好」「你在吗」「你能做什么」这类简单闲聊/功能咨询，直接走规则+降级逻辑，减少一次大模型调用
        - 对明显是方案/运动会/课课练等复杂需求，仍然走完整的深度分析
        """
        text = (user_input or "").strip()
        if not text:
            return False

        lower = text.lower()

        # 明确的功能咨询 / 自我介绍类
        capability_kws = [
            "你能做什么", "你可以做什么", "你会什么", "你有什么功能",
            "你是做什么的", "你是谁", "介绍一下你", "功能介绍",
            "可以帮我干什么", "能干嘛", "能帮我做什么",
            "what can you do", "who are you", "your ability",
        ]
        if any(kw in text for kw in capability_kws) or any(kw in lower for kw in ["what can you do", "who are you"]):
            return True

        # 问候/寒暄类
        greeting_kws = [
            "你好", "您好", "在吗", "嗨", "哈喽", "早上好", "下午好", "晚上好",
            "hello", "hi", "hey"
        ]
        if any(kw in text for kw in greeting_kws):
            return True

        # 非常短的小句子且不包含明显的方案/训练/运动会等复杂需求关键词
        complex_kws = [
            "方案", "计划", "设计", "课课练", "运动会", "比赛", "训练", "分析",
            "全员", "全校", "备课"
        ]
        if len(text) <= 12 and not any(kw in text for kw in complex_kws):
            return True

        return False

    def _parse_thought_result(self, content: str) -> ThoughtResult:
        """解析思考结果"""
        try:
            # 查找JSON部分
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return ThoughtResult(
                    understood_intent=data.get("understood_intent", "chat"),
                    explicit_needs=data.get("explicit_needs", []),
                    implicit_needs=data.get("implicit_needs", []),
                    potential_needs=data.get("potential_needs", []),
                    constraints=data.get("constraints", []),
                    sub_goals=data.get("sub_goals", []),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    suggested_approach=data.get("suggested_approach", ""),
                    needs_clarification=data.get("needs_clarification", False),
                    clarification_questions=data.get("clarification_questions", [])
                )
        except Exception as e:
            logger.error(f"解析思考结果失败: {e}")
        
        return self._create_fallback_result("")

    def _create_fallback_result(self, user_input: str) -> ThoughtResult:
        """创建降级结果"""
        # 简单的关键词匹配作为降级方案
        intent = "chat"
        if any(kw in user_input for kw in ["运动会", "全员", "全校"]):
            intent = "sports_meeting"
        elif any(kw in user_input for kw in ["课课练", "备课", "教学", "训练"]):
            intent = "lesson_plan"

        return ThoughtResult(
            understood_intent=intent,
            explicit_needs=[user_input] if user_input else [],
            implicit_needs=[],
            potential_needs=[],
            constraints=[],
            sub_goals=[],
            confidence=0.3,
            reasoning="降级处理：使用关键词匹配",
            suggested_approach="direct_response",
            needs_clarification=False,
            clarification_questions=[]
        )

    def create_user_goal(self, thought_result: ThoughtResult) -> UserGoal:
        """从思考结果创建用户目标"""
        return UserGoal(
            explicit_goal=thought_result.understood_intent,
            implicit_goals=thought_result.implicit_needs,
            constraints=thought_result.constraints,
            priority=5,
            confidence=thought_result.confidence
        )

    def should_ask_clarification(self, thought_result: ThoughtResult) -> bool:
        """判断是否需要向用户澄清"""
        # 置信度低于阈值时需要澄清
        if thought_result.confidence < 0.5:
            return True
        # 明确标记需要澄清
        if thought_result.needs_clarification:
            return True
        return False

    def generate_clarification(
        self,
        thought_result: ThoughtResult,
        context: Dict[str, Any]
    ) -> str:
        """生成澄清问题"""
        if thought_result.clarification_questions:
            return thought_result.clarification_questions[0]

        # 根据意图生成默认澄清问题
        intent = thought_result.understood_intent
        if intent == "sports_meeting":
            return "老师您好！请问您是想设计全员运动会方案吗？能告诉我大概有多少学生参加，以及场地条件如何吗？"
        elif intent == "lesson_plan":
            return "老师您好！请问您是想设计课课练方案吗？能告诉我是哪个年级的学生，以及想针对哪方面进行训练吗？"
        else:
            return "老师您好！请问有什么可以帮您的吗？"

    def enrich_understanding(
        self,
        thought_result: ThoughtResult,
        new_info: Dict[str, Any]
    ) -> ThoughtResult:
        """根据新信息丰富理解"""
        # 更新置信度
        new_confidence = min(1.0, thought_result.confidence + 0.2)

        # 合并新信息到显性需求
        new_explicit = thought_result.explicit_needs.copy()
        for key, value in new_info.items():
            if value:
                new_explicit.append(f"{key}: {value}")

        return ThoughtResult(
            understood_intent=thought_result.understood_intent,
            explicit_needs=new_explicit,
            implicit_needs=thought_result.implicit_needs,
            potential_needs=thought_result.potential_needs,
            constraints=thought_result.constraints,
            sub_goals=thought_result.sub_goals,
            confidence=new_confidence,
            reasoning=thought_result.reasoning + " [已更新]",
            suggested_approach=thought_result.suggested_approach,
            needs_clarification=False,
            clarification_questions=[]
        )

