#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能Agent核心 - 目标导向的智能决策系统

核心理念：
1. 从流程导向到目标导向
2. 从确定执行到动态决策
3. 从工具到伙伴

智能循环：
感知 → 理解 → 规划 → 执行 → 反思 → 学习
"""

import json
import logging
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_version.agent.memory import AgentMemory, TaskState, UserGoal
from agent_version.agent.thinking import ThinkingEngine, ThoughtResult
from agent_version.agent.strategist import Strategist, Strategy
from agent_version.agent.executor import Executor, ExecutionStatus
from agent_version.agent.reflector import Reflector
from agent_version.tools import (
    IntentRecognitionTool,
    EntityExtractionTool,
    LessonPlanSearchTool,
    SportsMeetingSearchTool,
    ClassProfileLookupTool,
    ClassDataAnalysisTool,
)

logger = logging.getLogger("agent")


class TeacherAgent:
    """
    智能体育教师备课助手Agent

    核心能力：
    1. 深度理解用户意图（显性、隐性、潜在需求）
    2. 动态生成多种策略并选择最优
    3. 弹性执行并实时调整
    4. 反思学习持续优化

    智能循环：感知 → 理解 → 规划 → 执行 → 反思
    """

    def __init__(self, session_id: str = "default"):
        """初始化智能Agent"""
        self.session_id = session_id

        # 核心组件
        self.memory = AgentMemory()
        self.thinking_engine = ThinkingEngine()
        self.strategist = Strategist()
        self.executor = Executor()
        self.reflector = Reflector(self.memory)

        # 工具集（用于辅助功能）
        self._init_tools()

        logger.info(f"智能Agent初始化完成: session_id={session_id}")

    def _init_tools(self):
        """初始化辅助工具"""
        self.intent_tool = IntentRecognitionTool()
        self.entity_tool = EntityExtractionTool()
        self.lesson_search_tool = LessonPlanSearchTool()
        self.sports_search_tool = SportsMeetingSearchTool()
        self.class_lookup_tool = ClassProfileLookupTool()
        self.class_analysis_tool = ClassDataAnalysisTool()

    def chat(self, user_input: str) -> str:
        """
        智能处理用户输入

        智能循环：感知 → 理解 → 规划 → 执行 → 反思
        """
        start_time = datetime.now().isoformat()

        # 1. 感知阶段：记录输入，获取上下文
        self.memory.add_message(self.session_id, "user", user_input)
        context = self.memory.get_full_context(self.session_id)
        self.memory.update_task_state(self.session_id, TaskState.ANALYZING)

        try:
            # 2. 理解阶段：深度分析用户意图
            thought_result = self.thinking_engine.analyze_request(user_input, context)
            logger.info(f"[理解] 意图={thought_result.understood_intent}, 置信度={thought_result.confidence}")

            # 设置用户目标
            user_goal = self.thinking_engine.create_user_goal(thought_result)
            self.memory.set_current_goal(self.session_id, user_goal)

            # 3. 规划阶段：生成策略并选择最优
            self.memory.update_task_state(self.session_id, TaskState.PLANNING)
            response = self._plan_and_execute(user_input, thought_result, context, start_time)

            # 记录回复
            self.memory.add_message(self.session_id, "assistant", response)
            self.memory.update_task_state(self.session_id, TaskState.COMPLETED)

            return response

        except Exception as e:
            logger.error(f"智能Agent执行失败: {e}")
            self.memory.update_task_state(self.session_id, TaskState.FAILED)
            error_msg = "抱歉，处理过程中遇到了问题。请您换一种方式描述需求，或者稍后再试。"
            self.memory.add_message(self.session_id, "assistant", error_msg)
            return error_msg

    def chat_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        智能流式处理用户输入

        智能循环：感知 → 理解 → 规划 → 执行 → 反思
        """
        start_time = datetime.now().isoformat()

        # 1. 感知阶段
        self.memory.add_message(self.session_id, "user", user_input)
        context = self.memory.get_full_context(self.session_id)
        self.memory.update_task_state(self.session_id, TaskState.ANALYZING)

        try:
            # 2. 理解阶段
            thought_result = self.thinking_engine.analyze_request(user_input, context)
            logger.info(f"[理解] 意图={thought_result.understood_intent}, 置信度={thought_result.confidence}")

            user_goal = self.thinking_engine.create_user_goal(thought_result)
            self.memory.set_current_goal(self.session_id, user_goal)

            # 3. 规划和执行阶段（流式）
            self.memory.update_task_state(self.session_id, TaskState.PLANNING)
            full_response = ""

            for chunk in self._plan_and_execute_stream(user_input, thought_result, context, start_time):
                full_response += chunk
                yield chunk

            self.memory.add_message(self.session_id, "assistant", full_response)
            self.memory.update_task_state(self.session_id, TaskState.COMPLETED)

        except Exception as e:
            logger.error(f"流式处理失败: {e}")
            self.memory.update_task_state(self.session_id, TaskState.FAILED)
            error_msg = "抱歉，处理过程中遇到了问题。请您换一种方式描述需求，或者稍后再试。"
            yield error_msg
            self.memory.add_message(self.session_id, "assistant", error_msg)

    def _plan_and_execute(
        self,
        user_input: str,
        thought_result: ThoughtResult,
        context: Dict[str, Any],
        start_time: str
    ) -> str:
        """规划并执行（非流式）"""
        intent = thought_result.understood_intent

        # 提取参数
        params = self._extract_and_accumulate_params(user_input, intent, context)

        # 获取相关知识
        knowledge = self.memory.get_relevant_knowledge(intent)

        # 生成策略
        strategies = self.strategist.generate_strategies(thought_result, context, knowledge)
        evaluations = self.strategist.evaluate_strategies(strategies, context)
        best_strategy = self.strategist.select_best_strategy(strategies, evaluations)

        logger.info(f"[规划] 选择策略: {best_strategy.name}, 类型: {best_strategy.strategy_type.value}")

        # 执行阶段
        self.memory.update_task_state(self.session_id, TaskState.EXECUTING)

        # 检索内容（如果需要）
        search_results = None
        if best_strategy.name in ["直接生成方案", "完整方案生成", "智能推断生成"]:
            search_results = self._search_content(intent, params)

        # 执行策略
        params["user_input"] = user_input
        params["intent"] = intent
        exec_ctx = self.executor.execute(
            strategy=best_strategy,
            intent=intent,
            params=params,
            search_results=search_results,
            context=context
        )

        # 反思阶段
        self.memory.update_task_state(self.session_id, TaskState.REFLECTING)
        reflection = self.reflector.reflect(exec_ctx, context=context)
        logger.info(f"[反思] 效果评分: {reflection.effectiveness_score}, 成功: {reflection.success}")

        return exec_ctx.intermediate_output or "抱歉，无法生成回复。"

    def _plan_and_execute_stream(
        self,
        user_input: str,
        thought_result: ThoughtResult,
        context: Dict[str, Any],
        start_time: str
    ) -> Generator[str, None, None]:
        """规划并执行（流式）"""
        intent = thought_result.understood_intent

        # 提取参数
        params = self._extract_and_accumulate_params(user_input, intent, context)

        # 获取相关知识
        knowledge = self.memory.get_relevant_knowledge(intent)

        # 生成策略
        strategies = self.strategist.generate_strategies(thought_result, context, knowledge)
        evaluations = self.strategist.evaluate_strategies(strategies, context)
        best_strategy = self.strategist.select_best_strategy(strategies, evaluations)

        logger.info(f"[规划] 选择策略: {best_strategy.name}")

        # 执行阶段
        self.memory.update_task_state(self.session_id, TaskState.EXECUTING)

        # 检索内容
        search_results = None
        if best_strategy.name in ["直接生成方案", "完整方案生成", "智能推断生成"]:
            search_results = self._search_content(intent, params)

        # 流式执行
        params["user_input"] = user_input
        params["intent"] = intent
        for chunk in self.executor.execute_stream(
            strategy=best_strategy,
            intent=intent,
            params=params,
            search_results=search_results,
            context=context
        ):
            yield chunk

    def _extract_and_accumulate_params(
        self,
        user_input: str,
        intent: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """提取并累积参数"""
        history = context.get("conversation_history", [])

        # 使用实体提取工具
        entity_result = json.loads(self.entity_tool._run(
            user_text=user_input,
            plan_type=intent,
            conversation_history=history
        ))
        entity_data = entity_result.get("data", {}) or {}
        params = entity_data.get("params", {}) or {}

        # 将实体工具识别出的缺失字段与计划类型也一起保存，便于后续引导或生成阶段使用
        missing = entity_data.get("missing") or []
        detected_plan_type = entity_data.get("plan_type")
        if missing:
            params["_missing_fields"] = missing
        if detected_plan_type:
            params["_detected_plan_type"] = detected_plan_type

        # 更新累积参数
        self.memory.update_accumulated_params(intent, params)
        accumulated = self.memory.get_accumulated_params(intent)
        accumulated.update(params)

        return accumulated

    def _search_content(self, intent: str, params: Dict) -> Optional[List[Dict[str, Any]]]:
        """检索相关内容

        返回值为结构化结果列表，方便后续在方案生成阶段灵活利用。
        """
        try:
            if intent == "lesson_plan":
                # 使用原项目的参数名
                grades = params.get("grades_query", "")
                weaknesses = params.get("trained_weaknesses", "")
                result = json.loads(self.lesson_search_tool._run(
                    grades_query=grades,
                    trained_weaknesses=weaknesses,
                    semantic_query=weaknesses if weaknesses else "体能训练",
                    top_k=10
                ))
                data = result.get("data", [])
                if isinstance(data, list) and data:
                    return data
            elif intent == "sports_meeting":
                # 使用原项目的参数名
                semantic = params.get("semantic_query", "运动会项目")
                grades = params.get("grades_query", "")
                count = params.get("count_query", "")
                result = json.loads(self.sports_search_tool._run(
                    semantic_query=semantic,
                    grades_query=grades,
                    count_query=str(count) if count else "",
                    top_k=5
                ))
                data = result.get("data", [])
                if isinstance(data, list) and data:
                    return data
        except Exception as e:
            logger.warning(f"检索失败: {e}")

        return None

    def _format_search_results(self, results: List[Dict]) -> str:
        """格式化检索结果"""
        formatted = []
        for i, item in enumerate(results[:5], 1):
            content = item.get("content", item.get("text", ""))
            if content:
                formatted.append(f"【参考{i}】\n{content[:500]}")
        return "\n\n".join(formatted) if formatted else ""

    def reset(self):
        """重置Agent状态"""
        self.memory.reset(self.session_id)
        logger.info(f"Agent已重置: session_id={self.session_id}")

    def get_session_info(self) -> Dict[str, Any]:
        """获取当前会话信息"""
        return self.memory.get_session_info(self.session_id)
