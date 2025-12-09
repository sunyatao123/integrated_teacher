#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略生成器 - 多策略生成与评估选择

实现：
1. 多策略生成：为每个任务生成多种执行策略
2. 多维度评估：成功率、效率、用户体验、风险
3. 智能选择：根据上下文选择最优策略
4. 不确定性处理：为每种策略添加置信度
"""

import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from agent_version.agent.thinking import ThoughtResult
from agent_version.tools.base import AIModelClient

logger = logging.getLogger("agent_strategist")


class StrategyType(Enum):
    """策略类型"""
    CONSERVATIVE = "conservative"  # 保守策略：低风险，稳定
    OPTIMIZED = "optimized"  # 优化策略：高效率
    INNOVATIVE = "innovative"  # 创新策略：高回报
    ADAPTIVE = "adaptive"  # 自适应策略：根据反馈调整


@dataclass
class Strategy:
    """策略定义"""
    id: str
    name: str
    strategy_type: StrategyType
    description: str
    steps: List[str]  # 执行步骤
    required_params: List[str]  # 必需参数
    optional_params: List[str]  # 可选参数
    success_probability: float  # 成功概率
    efficiency_score: float  # 效率评分
    user_experience_score: float  # 用户体验评分
    risk_level: float  # 风险等级
    confidence: float  # 置信度
    rationale: str  # 选择理由


@dataclass
class StrategyEvaluation:
    """策略评估结果"""
    strategy_id: str
    overall_score: float
    dimension_scores: Dict[str, float]
    pros: List[str]
    cons: List[str]
    recommendation: str


class Strategist:
    """
    策略生成器
    
    核心能力：
    1. 根据目标生成多种策略
    2. 多维度评估策略
    3. 智能选择最优策略
    4. 支持策略动态调整
    """
    
    def __init__(self):
        self.client = AIModelClient()
    
    def generate_strategies(
        self,
        thought_result: ThoughtResult,
        context: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> List[Strategy]:
        """生成多种策略
        
        参数:
            thought_result: 思考结果
            context: 上下文信息
            knowledge: 相关知识（经验、最佳实践）
        """
        raw_intent = (thought_result.understood_intent or "chat").strip()
        intent = self._normalize_intent(raw_intent)

        # 根据意图类型生成策略（先做意图归一化，尽量避免落入模板化的默认策略）
        if intent == "lesson_plan":
            return self._generate_lesson_plan_strategies(thought_result, context, knowledge)
        elif intent == "sports_meeting":
            return self._generate_sports_meeting_strategies(thought_result, context, knowledge)
        elif intent == "chat":
            # 闲聊类统一走智能对话策略，但在内部根据最近一轮用户输入区分
            # 问候 / 功能咨询 / 一般对话 三种模式
            return self._generate_chat_strategies(thought_result, context)
        else:
            # 其他未知意图全部降级为智能闲聊，而不是直接走死板的默认模板
            return self._generate_chat_strategies(thought_result, context)

    def _normalize_intent(self, intent: str) -> str:
        """对思考引擎返回的意图做归一化，兼容中英文/别名

        主要目标是：
        - 将各种"打招呼"、"功能介绍"等同义表达归到 chat
        - 将含有"课课练/教学/训练"的意图归到 lesson_plan
        - 将含有"运动会/比赛"的意图归到 sports_meeting
        """
        if not intent:
            return "chat"

        norm = intent.strip().lower()

        # 精确别名映射
        alias_map = {
            "greeting": "chat",
            "hello": "chat",
            "hi": "chat",
            "salute": "chat",
            "self_introduction": "chat",
            "self-introduction": "chat",
            "introduction": "chat",
            "capability": "chat",
            "capabilities": "chat",
            "what_can_you_do": "chat",
            "clarification": "chat",  # 统一走闲聊链路，由提示词引导澄清
        }
        if norm in alias_map:
            return alias_map[norm]

        # 中文关键字判断
        if any(kw in intent for kw in ["你好", "您好", "嗨", "在吗", "早上好", "下午好", "晚上好", "打招呼", "问候"]):
            return "chat"
        if any(kw in intent for kw in ["聊天", "闲聊", "对话"]):
            return "chat"
        if any(kw in intent for kw in ["能做什么", "可以干什么", "会干啥", "功能介绍", "能力介绍", "你是谁", "介绍一下你"]):
            return "chat"

        if any(kw in intent for kw in ["运动会", "全员运动会", "比赛", "sports_meeting", "sports"]):
            return "sports_meeting"
        if any(kw in intent for kw in ["课课练", "备课", "教学", "训练", "lesson_plan", "lesson"]):
            return "lesson_plan"

        return norm or "chat"
    
    def _generate_lesson_plan_strategies(
        self,
        thought_result: ThoughtResult,
        context: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> List[Strategy]:
        """生成课课练方案策略"""
        strategies = []
        accumulated = context.get("accumulated_params", {}).get("lesson_plan", {})

        # 检查参数完整性 - 使用原项目的参数名
        # 课课练：grades_query 或 trained_weaknesses 满足任一即可
        has_grade = bool(accumulated.get("grades_query"))
        has_weakness = bool(accumulated.get("trained_weaknesses"))

        # 只要有年级或薄弱项之一，就可以直接生成
        if has_grade or has_weakness:
            # 策略1：直接生成（参数足够）- 高优先级
            strategies.append(Strategy(
                id=str(uuid.uuid4())[:8],
                name="直接生成方案",
                strategy_type=StrategyType.OPTIMIZED,
                description="参数已足够，直接检索并生成课课练方案",
                steps=["检索相关教学资料", "生成个性化方案", "输出完整方案"],
                required_params=["grades_query"] if has_grade else ["trained_weaknesses"],
                optional_params=["trained_weaknesses"] if has_grade else ["grades_query"],
                success_probability=0.95,  # 提高成功率
                efficiency_score=0.95,
                user_experience_score=0.9,  # 提高用户体验分
                risk_level=0.1,
                confidence=0.95,  # 提高置信度
                rationale="已有足够参数，可以直接生成高质量方案"
            ))
        else:
            # 策略2：引导收集信息 - 只有在没有任何参数时才使用
            strategies.append(Strategy(
                id=str(uuid.uuid4())[:8],
                name="分步引导收集",
                strategy_type=StrategyType.CONSERVATIVE,
                description="引导用户提供年级或薄弱项信息",
                steps=["生成友好的引导语", "等待用户补充", "收集参数后再生成"],
                required_params=[],
                optional_params=["grades_query", "trained_weaknesses"],
                success_probability=0.9,
                efficiency_score=0.7,
                user_experience_score=0.85,
                risk_level=0.05,
                confidence=0.9,
                rationale="缺少必要参数，需要引导用户补充"
            ))

        return strategies if strategies else [self._create_fallback_strategy("lesson_plan")]

    def _generate_sports_meeting_strategies(
        self,
        thought_result: ThoughtResult,
        context: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> List[Strategy]:
        """生成运动会方案策略"""
        strategies = []
        accumulated = context.get("accumulated_params", {}).get("sports_meeting", {})

        # 检查参数 - 使用原项目的参数名
        # 运动会：只要有 semantic_query（操场条件描述）就可以生成
        # semantic_query 通常包含：操场大小、跑道数量、场地类型等综合信息
        has_semantic = bool(accumulated.get("semantic_query"))
        has_count = bool(accumulated.get("count_query"))
        has_grade = bool(accumulated.get("grades_query"))

        # 只要有操场条件描述，就可以生成方案
        if has_semantic:
            strategies.append(Strategy(
                id=str(uuid.uuid4())[:8],
                name="完整方案生成",
                strategy_type=StrategyType.OPTIMIZED,
                description="场地条件已明确，生成完整运动会方案",
                steps=["检索运动会项目库", "根据场地条件筛选", "生成完整方案"],
                required_params=["semantic_query"],
                optional_params=["count_query", "grades_query"],
                success_probability=0.95,
                efficiency_score=0.95,
                user_experience_score=0.9,
                risk_level=0.05,
                confidence=0.95,
                rationale="场地条件已明确，可以生成高质量的运动会方案"
            ))
        else:
            # 只有在没有操场条件信息时才引导
            strategies.append(Strategy(
                id=str(uuid.uuid4())[:8],
                name="分步引导收集",
                strategy_type=StrategyType.CONSERVATIVE,
                description="引导用户提供操场条件信息",
                steps=["生成引导语", "收集场地参数", "确认后生成方案"],
                required_params=[],
                optional_params=["semantic_query", "count_query", "grades_query"],
                success_probability=0.9,
                efficiency_score=0.7,
                user_experience_score=0.85,
                risk_level=0.1,
                confidence=0.9,
                rationale="需要了解操场条件才能设计合适的运动会项目"
            ))

        return strategies if strategies else [self._create_fallback_strategy("sports_meeting")]

    def _generate_chat_strategies(
        self,
        thought_result: ThoughtResult,
        context: Dict[str, Any]
    ) -> List[Strategy]:
        """生成闲聊策略

        根据最近一轮用户输入，将闲聊细分为：
        - 问候寒暄：更注重轻量回应 + 轻度引导
        - 功能咨询：重点清晰介绍能力与可立即尝试的示例
        - 一般对话：结合上下文进行自然交流
        """
        history = context.get("conversation_history", []) if context else []
        last_user_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "") or ""
                break

        chat_mode = self._detect_chat_mode(last_user_msg)

        if chat_mode == "greeting":
            name = "智能问候对话"
            description = "针对老师的打招呼或寒暄进行友好回应，并轻量引导到具体需求"
            steps = ["友好回应问候", "简要介绍自己能做什么", "轻度引导老师说明需求"]
        elif chat_mode == "capability":
            name = "能力介绍与示例"
            description = "系统介绍自身能力，并给出2-3个可立即尝试的具体示例"
            steps = ["识别老师在询问能力", "结构化列出主要能力", "结合场景给出可尝试示例"]
        else:
            name = "智能对话"
            description = "结合上下文进行自然对话，逐步澄清并锁定老师的核心目标"
            steps = ["参考最近对话历史", "自然回答老师问题", "在合适时机引导到关键场景（课课练/运动会）"]

        return [Strategy(
            id=str(uuid.uuid4())[:8],
            name=name,
            strategy_type=StrategyType.ADAPTIVE,
            description=description,
            steps=steps,
            required_params=[],
            optional_params=[],
            success_probability=0.95,
            efficiency_score=0.9,
            user_experience_score=0.96,
            risk_level=0.05,
            confidence=0.9,
            rationale="根据老师的语言风格自动选择问候/能力介绍/一般闲聊模式"
        )]

    def _detect_chat_mode(self, user_text: str) -> str:
        """根据用户最近一轮输入识别闲聊子类型

        返回值："greeting" / "capability" / "general"
        """
        if not user_text:
            return "general"

        text = user_text.strip().lower()

        # 功能咨询优先级最高（例如："你好，你能做什么？"）
        capability_kws = [
            "你能做什么", "你可以做什么", "你可以干什么", "你会干啥",
            "能帮我做什么", "你会什么", "功能", "功能介绍",
            "你是什么", "你是谁", "介绍一下你", "what can you do", "what can u do"
        ]
        if any(kw in user_text for kw in capability_kws):
            return "capability"

        greeting_kws = [
            "你好", "您好", "在吗", "嗨", "早上好", "下午好", "晚上好",
            "hello", "hi", "hey"
        ]
        if any(kw.lower() in text for kw in greeting_kws):
            return "greeting"

        return "general"

    def _generate_default_strategies(
        self,
        thought_result: ThoughtResult
    ) -> List[Strategy]:
        """生成默认策略"""
        return [self._create_fallback_strategy("default")]

    def _create_fallback_strategy(self, intent: str) -> Strategy:
        """创建降级策略"""
        return Strategy(
            id=str(uuid.uuid4())[:8],
            name="安全降级",
            strategy_type=StrategyType.CONSERVATIVE,
            description="使用安全的默认处理方式",
            steps=["确认用户意图", "提供基础帮助"],
            required_params=[],
            optional_params=[],
            success_probability=0.8,
            efficiency_score=0.6,
            user_experience_score=0.7,
            risk_level=0.1,
            confidence=0.7,
            rationale="降级策略，确保系统稳定运行"
        )

    def evaluate_strategies(
        self,
        strategies: List[Strategy],
        context: Dict[str, Any]
    ) -> List[StrategyEvaluation]:
        """评估所有策略"""
        evaluations = []

        for strategy in strategies:
            # 计算综合评分
            overall = (
                strategy.success_probability * 0.3 +
                strategy.efficiency_score * 0.25 +
                strategy.user_experience_score * 0.25 +
                (1 - strategy.risk_level) * 0.2
            )

            evaluations.append(StrategyEvaluation(
                strategy_id=strategy.id,
                overall_score=overall,
                dimension_scores={
                    "success": strategy.success_probability,
                    "efficiency": strategy.efficiency_score,
                    "experience": strategy.user_experience_score,
                    "safety": 1 - strategy.risk_level
                },
                pros=self._identify_pros(strategy),
                cons=self._identify_cons(strategy),
                recommendation=self._generate_recommendation(strategy, overall)
            ))

        # 按综合评分排序
        evaluations.sort(key=lambda x: x.overall_score, reverse=True)
        return evaluations

    def select_best_strategy(
        self,
        strategies: List[Strategy],
        evaluations: List[StrategyEvaluation],
        preference: Optional[str] = None
    ) -> Strategy:
        """选择最佳策略"""
        if not strategies:
            return self._create_fallback_strategy("default")

        # 如果有偏好，优先选择匹配的策略类型
        if preference:
            pref_map = {
                "safe": StrategyType.CONSERVATIVE,
                "fast": StrategyType.OPTIMIZED,
                "smart": StrategyType.INNOVATIVE
            }
            pref_type = pref_map.get(preference)
            if pref_type:
                for s in strategies:
                    if s.strategy_type == pref_type:
                        return s

        # 否则选择评分最高的
        if evaluations:
            best_id = evaluations[0].strategy_id
            for s in strategies:
                if s.id == best_id:
                    return s

        return strategies[0]

    def _identify_pros(self, strategy: Strategy) -> List[str]:
        """识别策略优点"""
        pros = []
        if strategy.success_probability > 0.8:
            pros.append("成功率高")
        if strategy.efficiency_score > 0.8:
            pros.append("执行效率高")
        if strategy.user_experience_score > 0.8:
            pros.append("用户体验好")
        if strategy.risk_level < 0.2:
            pros.append("风险低")
        return pros if pros else ["稳定可靠"]

    def _identify_cons(self, strategy: Strategy) -> List[str]:
        """识别策略缺点"""
        cons = []
        if strategy.success_probability < 0.7:
            cons.append("成功率较低")
        if strategy.efficiency_score < 0.7:
            cons.append("效率较低")
        if strategy.risk_level > 0.3:
            cons.append("存在一定风险")
        return cons

    def _generate_recommendation(self, strategy: Strategy, score: float) -> str:
        """生成推荐说明"""
        if score > 0.85:
            return "强烈推荐"
        elif score > 0.7:
            return "推荐使用"
        elif score > 0.5:
            return "可以考虑"
        else:
            return "谨慎使用"

