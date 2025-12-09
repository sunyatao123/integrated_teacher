#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行器 - 弹性执行与动态调整

实现：
1. 过程监控：监控执行过程指标
2. 动态调整：遇到障碍时重新评估和调整
3. 弹性执行：支持中途调整策略
4. 流式输出：支持实时流式响应
"""

import json
import logging
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agent_version.agent.strategist import Strategy, StrategyType
from agent_version.agent.memory import ExecutionRecord, TaskState
from agent_version.tools.base import AIModelClient, load_prompt_template
from agent_version.tools.generation_tool import PlanGenerationTool

logger = logging.getLogger("agent_executor")


class ExecutionStatus(Enum):
    """执行状态"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    ADJUSTED = "adjusted"


@dataclass
class ExecutionContext:
    """执行上下文"""
    strategy: Strategy
    params: Dict[str, Any]
    # 为了支持更智能的方案生成，这里改为保存结构化检索结果列表
    search_results: Optional[List[Dict[str, Any]]] = None
    intermediate_output: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    error: Optional[str] = None
    adjustments: List[str] = field(default_factory=list)


class Executor:
    """
    执行器
    
    核心能力：
    1. 执行选定的策略
    2. 监控执行过程
    3. 动态调整执行
    4. 支持流式输出
    """
    
    def __init__(self):
        self.client = AIModelClient()
        # 复用已有的方案生成工具，避免两套提示词逻辑导致风格不一致
        self.plan_tool = PlanGenerationTool()
        self._load_prompts()
    
    def _load_prompts(self):
        """加载提示词模板"""
        self.lesson_plan_prompt = load_prompt_template("plan_generation_lesson_plan.txt")
        self.sports_meeting_prompt = load_prompt_template("plan_generation_sports_meeting.txt")
        self.guidance_prompt = load_prompt_template("guidance_prompt.txt")
    
    def execute(
        self,
        strategy: Strategy,
        intent: str,
        params: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionContext:
        """
        执行策略（非流式）
        
        参数:
            strategy: 选定的策略
            intent: 意图类型
            params: 执行参数
            search_results: 检索结果
            context: 上下文信息
        """
        exec_ctx = ExecutionContext(
            strategy=strategy,
            params=params,
            search_results=search_results,
            status=ExecutionStatus.IN_PROGRESS
        )
        
        try:
            # 根据策略类型执行
            if strategy.name == "引导补充信息" or strategy.name == "分步引导收集":
                result = self._execute_guidance(intent, params, context)
            elif strategy.name in ["直接生成方案", "完整方案生成", "智能推断生成"]:
                result = self._execute_generation(intent, params, search_results, context)
            elif strategy.name in ["智能对话", "智能问候对话", "能力介绍与示例"]:
                chat_mode = "general"
                if strategy.name == "智能问候对话":
                    chat_mode = "greeting"
                elif strategy.name == "能力介绍与示例":
                    chat_mode = "capability"
                result = self._execute_chat(params.get("user_input", ""), context, chat_mode)
            else:
                result = self._execute_default(intent, params, context)
            
            exec_ctx.intermediate_output = result
            exec_ctx.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"执行失败: {e}")
            exec_ctx.error = str(e)
            exec_ctx.status = ExecutionStatus.FAILED
            exec_ctx = self._handle_failure(exec_ctx, context)
        
        return exec_ctx
    
    def execute_stream(
        self,
        strategy: Strategy,
        intent: str,
        params: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        """
        执行策略（流式）
        """
        try:
            # 根据策略类型执行
            if strategy.name == "引导补充信息" or strategy.name == "分步引导收集":
                result = self._execute_guidance(intent, params, context)
                yield result
            elif strategy.name in ["直接生成方案", "完整方案生成", "智能推断生成"]:
                yield from self._execute_generation_stream(intent, params, search_results, context)
            elif strategy.name in ["智能对话", "智能问候对话", "能力介绍与示例"]:
                chat_mode = "general"
                if strategy.name == "智能问候对话":
                    chat_mode = "greeting"
                elif strategy.name == "能力介绍与示例":
                    chat_mode = "capability"
                yield from self._execute_chat_stream(params.get("user_input", ""), context, chat_mode)
            else:
                result = self._execute_default(intent, params, context)
                yield result
                
        except Exception as e:
            logger.error(f"流式执行失败: {e}")
            yield f"抱歉，执行过程中遇到问题：{str(e)}"
    
    def _execute_guidance(
        self,
        intent: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """执行引导收集"""
        # 确定缺失的参数
        missing = self._get_missing_params(intent, params)
        
        prompt = self.guidance_prompt.format(
            user_text=params.get("user_input", ""),
            collected_info=json.dumps(params, ensure_ascii=False),
            plan_type=intent,
            missing_info=", ".join(missing) if missing else "无"
        )
        
        response = self.client.client.chat.completions.create(
            model=self.client.model,
            messages=[
                {"role": "system", "content": "你是一个友好的体育教学助手，正在帮助老师收集信息。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def _execute_generation(
        self,
        intent: str,
        params: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """执行方案生成（非流式）

        这里直接复用 PlanGenerationTool 的生成逻辑，保证与工具链提示词保持一致，
        同时利用其对检索结果和参数的结构化处理能力。
        """
        conversation_history: List[Dict[str, Any]] = []
        if context:
            conversation_history = context.get("conversation_history", []) or []

        user_text = params.get("user_input", "")
        results = search_results or []

        return self.plan_tool._generate_plan(
            plan_type=intent,
            user_text=user_text,
            params=params,
            results=results,
            conversation_history=conversation_history,
            need_guidance=False,
            missing=[]
        )

    def _execute_generation_stream(
        self,
        intent: str,
        params: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        """执行方案生成（流式）——复用 PlanGenerationTool 的流式生成能力"""
        conversation_history: List[Dict[str, Any]] = []
        if context:
            conversation_history = context.get("conversation_history", []) or []

        user_text = params.get("user_input", "")
        results = search_results or []

        for chunk in self.plan_tool._generate_plan_stream(
            plan_type=intent,
            user_text=user_text,
            params=params,
            results=results,
            conversation_history=conversation_history,
            need_guidance=False,
            missing=[]
        ):
            yield chunk

    def _execute_chat(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        chat_mode: str = "general"
    ) -> str:
        """执行闲聊（非流式）"""
        messages = self._build_chat_messages(user_input, context, chat_mode)

        response = self.client.client.chat.completions.create(
            model=self.client.model,
            messages=messages,
            max_tokens=1000,
            temperature=0.8
        )

        return response.choices[0].message.content.strip()

    def _execute_chat_stream(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        chat_mode: str = "general"
    ) -> Generator[str, None, None]:
        """执行闲聊（流式）"""
        messages = self._build_chat_messages(user_input, context, chat_mode)

        response = self.client.client.chat.completions.create(
            model=self.client.model,
            messages=messages,
            max_tokens=1000,
            temperature=0.8,
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _execute_default(
        self,
        intent: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """默认执行"""
        return "老师您好！请问有什么可以帮您的吗？我可以帮您设计课课练方案或全员运动会方案。"

    def _build_generation_prompt(
        self,
        intent: str,
        params: Dict[str, Any],
        search_results: Optional[str] = None
    ) -> str:
        """构建生成提示词"""
        if intent == "lesson_plan":
            template = self.lesson_plan_prompt
            return template.format(
                grade=params.get("grade", "未指定"),
                weakness=params.get("weakness", "综合体能"),
                search_results=search_results or "无检索结果"
            )
        elif intent == "sports_meeting":
            template = self.sports_meeting_prompt
            return template.format(
                student_count=params.get("student_count", "未指定"),
                grade=params.get("grade", "未指定"),
                field_type=params.get("field_type", "标准操场"),
                track_count=params.get("track_count", "4"),
                search_results=search_results or "无检索结果"
            )
        else:
            return f"请根据以下信息生成回复：{json.dumps(params, ensure_ascii=False)}"

    def _build_chat_messages(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        chat_mode: str = "general"
    ) -> List[Dict[str, Any]]:
        """根据聊天模式构建带有上下文的消息列表

        chat_mode: "greeting" / "capability" / "general"
        """
        system_prompt = load_prompt_template("teacher_system_prompt.txt")
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # 添加最近几轮历史对话，帮助大模型理解上下文，避免重复说同一句话
        if context:
            history = context.get("conversation_history", [])[-6:]
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if not content:
                    continue
                messages.append({"role": role, "content": content})

        if not user_input:
            user_input = "老师您好。"

        # 不同聊天模式下注入不同的行为指令，减少模板化感
        if chat_mode == "greeting":
            mode_instruction = (
                "【当前场景：老师只是友好地打招呼或寒暄。\n"
                "1. 先用自然、简短的方式回应问候；\n"
                "2. 再用 1-2 句话轻量介绍你能做什么；\n"
                "3. 最后可以礼貌地问一句老师目前最想解决的体育教学问题是什么。\n"
                "请用口语化、自然的语气回答，不要生硬罗列功能清单。】\n\n"
            )
        elif chat_mode == "capability":
            mode_instruction = (
                "【当前场景：老师在询问“你能做什么/有什么功能”。\n"
                "请用结构化方式简要介绍你的主要能力（例如：设计课课练方案、生成全员运动会方案、分析班级体测数据等），\n"
                "并给出 2-3 个老师可以立即尝试的具体指令示例。\n"
                "示例要贴近体育老师的真实场景，不要回答得过于模板化。】\n\n"
            )
        else:
            mode_instruction = (
                "【当前场景：普通对话/追问。\n"
                "请结合已有对话历史继续交流，避免重复之前说过的话，\n"
                "在合适的时候帮助老师逐步澄清和锁定自己的目标（比如是想要课课练方案，还是运动会设计）。】\n\n"
            )

        messages.append({
            "role": "user",
            "content": mode_instruction + user_input
        })

        return messages

    def _get_missing_params(self, intent: str, params: Dict[str, Any]) -> List[str]:
        """获取缺失的参数"""
        missing = []

        if intent == "lesson_plan":
            if not params.get("grade") and not params.get("weakness"):
                missing.extend(["年级", "薄弱项（速度/力量/柔韧/耐力等）"])
        elif intent == "sports_meeting":
            if not params.get("student_count"):
                missing.append("学生人数")
            if not params.get("grade"):
                missing.append("年级")
            if not params.get("field_type"):
                missing.append("场地类型")
            if not params.get("track_count"):
                missing.append("跑道数量")

        return missing

    def _handle_failure(
        self,
        exec_ctx: ExecutionContext,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionContext:
        """处理执行失败"""
        # 记录调整
        exec_ctx.adjustments.append(f"执行失败: {exec_ctx.error}")

        # 尝试降级处理
        exec_ctx.intermediate_output = "抱歉，处理过程中遇到了一些问题。请您稍后再试，或者换一种方式描述您的需求。"
        exec_ctx.status = ExecutionStatus.ADJUSTED

        return exec_ctx

    def create_execution_record(
        self,
        exec_ctx: ExecutionContext,
        start_time: str
    ) -> ExecutionRecord:
        """创建执行记录"""
        return ExecutionRecord(
            strategy_id=exec_ctx.strategy.id,
            strategy_name=exec_ctx.strategy.name,
            start_time=start_time,
            end_time=datetime.now().isoformat(),
            success=exec_ctx.status == ExecutionStatus.COMPLETED,
            result=exec_ctx.intermediate_output[:500] if exec_ctx.intermediate_output else None,
            feedback=None,
            lessons_learned=exec_ctx.adjustments
        )

