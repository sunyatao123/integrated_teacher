#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反思器 - 经验总结与持续学习

实现：
1. 经验总结：系统性总结成功和失败经验
2. 知识积累：将经验转化为可复用知识
3. 持续优化：改进决策模型和执行策略
4. 反馈学习：从用户反馈中学习
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from agent_version.agent.memory import Experience, ExecutionRecord, AgentMemory
from agent_version.agent.strategist import Strategy
from agent_version.agent.executor import ExecutionContext, ExecutionStatus
from agent_version.tools.base import AIModelClient

logger = logging.getLogger("agent_reflector")


@dataclass
class ReflectionResult:
    """反思结果"""
    success: bool
    effectiveness_score: float  # 0-1
    key_insights: List[str]  # 关键洞察
    lessons_learned: List[str]  # 经验教训
    improvement_suggestions: List[str]  # 改进建议
    should_update_knowledge: bool  # 是否应更新知识库


class Reflector:
    """
    反思器
    
    核心能力：
    1. 分析执行结果
    2. 总结经验教训
    3. 更新知识库
    4. 提供改进建议
    """
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.client = AIModelClient()
    
    def reflect(
        self,
        exec_ctx: ExecutionContext,
        user_feedback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReflectionResult:
        """
        反思执行结果
        
        参数:
            exec_ctx: 执行上下文
            user_feedback: 用户反馈（如果有）
            context: 上下文信息
        """
        # 基础评估
        success = exec_ctx.status == ExecutionStatus.COMPLETED
        
        # 计算效果评分
        effectiveness = self._calculate_effectiveness(exec_ctx, user_feedback)
        
        # 生成洞察和教训
        insights, lessons = self._generate_insights(exec_ctx, user_feedback, context)
        
        # 生成改进建议
        suggestions = self._generate_suggestions(exec_ctx, effectiveness)
        
        # 判断是否应更新知识库
        should_update = effectiveness > 0.7 or (not success and effectiveness < 0.3)
        
        result = ReflectionResult(
            success=success,
            effectiveness_score=effectiveness,
            key_insights=insights,
            lessons_learned=lessons,
            improvement_suggestions=suggestions,
            should_update_knowledge=should_update
        )
        
        # 如果应该更新知识库，则记录经验
        if should_update:
            self._record_experience(exec_ctx, result)
        
        return result
    
    def _calculate_effectiveness(
        self,
        exec_ctx: ExecutionContext,
        user_feedback: Optional[str] = None
    ) -> float:
        """计算执行效果评分"""
        score = 0.5  # 基础分
        
        # 执行状态影响
        if exec_ctx.status == ExecutionStatus.COMPLETED:
            score += 0.3
        elif exec_ctx.status == ExecutionStatus.ADJUSTED:
            score += 0.1
        elif exec_ctx.status == ExecutionStatus.FAILED:
            score -= 0.3
        
        # 策略置信度影响
        score += exec_ctx.strategy.confidence * 0.1
        
        # 用户反馈影响
        if user_feedback:
            if any(kw in user_feedback for kw in ["好", "谢谢", "不错", "满意"]):
                score += 0.2
            elif any(kw in user_feedback for kw in ["不好", "不对", "错误", "重新"]):
                score -= 0.2
        
        # 确保在0-1范围内
        return max(0.0, min(1.0, score))
    
    def _generate_insights(
        self,
        exec_ctx: ExecutionContext,
        user_feedback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """生成洞察和教训"""
        insights = []
        lessons = []
        
        strategy = exec_ctx.strategy
        
        # 策略相关洞察
        if strategy.strategy_type.value == "conservative":
            if exec_ctx.status == ExecutionStatus.COMPLETED:
                insights.append("保守策略在此场景下表现良好")
            else:
                lessons.append("保守策略未能解决问题，可能需要更主动的方式")
        
        elif strategy.strategy_type.value == "optimized":
            if exec_ctx.status == ExecutionStatus.COMPLETED:
                insights.append("优化策略成功提高了效率")
            else:
                lessons.append("优化策略可能过于激进，需要更多验证")
        
        elif strategy.strategy_type.value == "innovative":
            if exec_ctx.status == ExecutionStatus.COMPLETED:
                insights.append("创新策略带来了更好的用户体验")
            else:
                lessons.append("创新策略风险较高，需要更好的降级方案")
        
        # 参数相关洞察
        params = exec_ctx.params
        if not params.get("grade") and not params.get("weakness"):
            lessons.append("缺少关键参数时应优先引导用户补充")
        
        # 调整相关洞察
        if exec_ctx.adjustments:
            lessons.append(f"执行过程中进行了{len(exec_ctx.adjustments)}次调整")
        
        return insights, lessons
    
    def _generate_suggestions(
        self,
        exec_ctx: ExecutionContext,
        effectiveness: float
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if effectiveness < 0.5:
            suggestions.append("考虑使用更保守的策略")
            suggestions.append("增加参数验证步骤")
        
        if exec_ctx.status == ExecutionStatus.FAILED:
            suggestions.append("增强错误处理机制")
            suggestions.append("添加更多的降级方案")
        
        if exec_ctx.adjustments:
            suggestions.append("优化初始策略选择，减少执行中调整")
        
        if not suggestions:
            suggestions.append("当前策略表现良好，可以继续使用")
        
        return suggestions
    
    def _record_experience(
        self,
        exec_ctx: ExecutionContext,
        result: ReflectionResult
    ):
        """记录经验到知识库"""
        # 确定场景类型
        params = exec_ctx.params
        if params.get("intent") == "lesson_plan":
            scenario_type = "lesson_plan"
        elif params.get("intent") == "sports_meeting":
            scenario_type = "sports_meeting"
        else:
            scenario_type = "general"
        
        experience = Experience(
            scenario_type=scenario_type,
            user_intent=params.get("user_input", "")[:100],
            strategy_used=exec_ctx.strategy.name,
            success=result.success,
            effectiveness_score=result.effectiveness_score,
            key_factors=result.key_insights,
            lessons=result.lessons_learned
        )
        
        self.memory.record_experience(experience)
        logger.info(f"记录经验: {scenario_type}, 效果评分: {result.effectiveness_score}")
    
    def get_relevant_lessons(
        self,
        scenario_type: str
    ) -> Dict[str, Any]:
        """获取相关经验教训"""
        knowledge = self.memory.get_relevant_knowledge(scenario_type)
        
        return {
            "best_practices": knowledge.get("best_practices", []),
            "failure_patterns": knowledge.get("failure_patterns", []),
            "recent_experiences": knowledge.get("experiences", [])[:3]
        }
    
    def suggest_strategy_adjustment(
        self,
        current_strategy: Strategy,
        exec_ctx: ExecutionContext
    ) -> Optional[str]:
        """建议策略调整"""
        if exec_ctx.status == ExecutionStatus.COMPLETED:
            return None
        
        # 根据当前策略类型建议调整
        if current_strategy.strategy_type.value == "innovative":
            return "建议切换到保守策略，确保基本功能正常"
        elif current_strategy.strategy_type.value == "optimized":
            return "建议增加验证步骤，确保参数完整"
        else:
            return "建议检查输入参数和系统状态"
