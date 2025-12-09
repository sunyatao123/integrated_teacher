#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能Agent记忆系统 - 增强版

实现多层记忆架构：
- 短期记忆：当前会话上下文
- 工作记忆：当前任务执行状态
- 长期记忆：用户偏好、历史方案
- 知识库：积累的经验和最佳实践
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("agent_memory")


class TaskState(Enum):
    """任务状态"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UserGoal:
    """用户目标表示"""
    explicit_goal: str  # 显性目标
    implicit_goals: List[str] = field(default_factory=list)  # 隐性目标
    constraints: List[str] = field(default_factory=list)  # 约束条件
    priority: int = 5  # 优先级 1-10
    confidence: float = 0.8  # 理解置信度


@dataclass
class ExecutionRecord:
    """执行记录"""
    strategy_id: str
    strategy_name: str
    start_time: str
    end_time: Optional[str] = None
    success: bool = False
    result: Optional[str] = None
    feedback: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class Experience:
    """经验记录"""
    scenario_type: str  # 场景类型
    user_intent: str  # 用户意图
    strategy_used: str  # 使用的策略
    success: bool  # 是否成功
    effectiveness_score: float  # 效果评分 0-1
    key_factors: List[str] = field(default_factory=list)  # 关键因素
    lessons: List[str] = field(default_factory=list)  # 经验教训
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class WorkingMemory:
    """工作记忆：管理当前任务的执行状态"""

    def __init__(self):
        self.current_goal: Optional[UserGoal] = None
        self.current_state: TaskState = TaskState.PENDING
        self.strategies: List[Dict[str, Any]] = []
        self.selected_strategy: Optional[Dict[str, Any]] = None
        self.execution_records: List[ExecutionRecord] = []
        self.intermediate_results: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}

    def set_goal(self, goal: UserGoal):
        """设置当前目标"""
        self.current_goal = goal
        self.current_state = TaskState.ANALYZING

    def add_strategy(self, strategy: Dict[str, Any]):
        """添加候选策略"""
        self.strategies.append(strategy)

    def select_strategy(self, strategy_id: str):
        """选择执行策略"""
        for s in self.strategies:
            if s.get("id") == strategy_id:
                self.selected_strategy = s
                self.current_state = TaskState.EXECUTING
                break

    def record_execution(self, record: ExecutionRecord):
        """记录执行结果"""
        self.execution_records.append(record)

    def update_context(self, key: str, value: Any):
        """更新上下文"""
        self.context[key] = value

    def reset(self):
        """重置工作记忆"""
        self.current_goal = None
        self.current_state = TaskState.PENDING
        self.strategies = []
        self.selected_strategy = None
        self.execution_records = []
        self.intermediate_results = {}
        self.context = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_goal": asdict(self.current_goal) if self.current_goal else None,
            "current_state": self.current_state.value,
            "strategies": self.strategies,
            "selected_strategy": self.selected_strategy,
            "execution_records": [asdict(r) for r in self.execution_records],
            "intermediate_results": self.intermediate_results,
            "context": self.context
        }


class KnowledgeBase:
    """知识库：积累的经验和最佳实践"""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.experiences: List[Experience] = []
        self.best_practices: Dict[str, List[str]] = {}
        self.failure_patterns: Dict[str, List[str]] = {}

        if storage_path and storage_path.exists():
            self._load()

    def add_experience(self, exp: Experience):
        """添加经验"""
        self.experiences.append(exp)

        # 更新最佳实践或失败模式
        if exp.success and exp.effectiveness_score > 0.7:
            if exp.scenario_type not in self.best_practices:
                self.best_practices[exp.scenario_type] = []
            self.best_practices[exp.scenario_type].extend(exp.lessons)
        elif not exp.success:
            if exp.scenario_type not in self.failure_patterns:
                self.failure_patterns[exp.scenario_type] = []
            self.failure_patterns[exp.scenario_type].extend(exp.lessons)

        self._save()

    def get_relevant_experiences(
        self,
        scenario_type: str,
        limit: int = 5
    ) -> List[Experience]:
        """获取相关经验"""
        relevant = [e for e in self.experiences if e.scenario_type == scenario_type]
        # 按效果评分排序
        relevant.sort(key=lambda x: x.effectiveness_score, reverse=True)
        return relevant[:limit]

    def get_best_practices(self, scenario_type: str) -> List[str]:
        """获取最佳实践"""
        return self.best_practices.get(scenario_type, [])

    def get_failure_patterns(self, scenario_type: str) -> List[str]:
        """获取失败模式"""
        return self.failure_patterns.get(scenario_type, [])

    def _save(self):
        """保存到文件"""
        if not self.storage_path:
            return

        data = {
            "experiences": [asdict(e) for e in self.experiences[-100:]],  # 保留最近100条
            "best_practices": self.best_practices,
            "failure_patterns": self.failure_patterns
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从文件加载"""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.experiences = [
                Experience(**e) for e in data.get("experiences", [])
            ]
            self.best_practices = data.get("best_practices", {})
            self.failure_patterns = data.get("failure_patterns", {})
        except Exception as e:
            logger.warning(f"加载知识库失败: {e}")


class AgentMemory:
    """
    智能Agent记忆系统 - 增强版

    多层记忆架构：
    1. 短期记忆：对话历史、当前参数
    2. 工作记忆：任务状态、执行上下文
    3. 长期记忆：用户偏好、历史方案
    4. 知识库：经验积累、最佳实践
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """初始化记忆系统"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = Path(__file__).parent.parent / "cache"
            self.cache_dir.mkdir(exist_ok=True)

        # 短期记忆（会话级）
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # 工作记忆（任务级）
        self.working_memories: Dict[str, WorkingMemory] = {}

        # 长期记忆：累积参数
        self.accumulated_lesson_plan: Dict[str, Any] = {}
        self.accumulated_sports_meeting: Dict[str, Any] = {}

        # 用户偏好
        self.user_preferences: Dict[str, Any] = {}

        # 知识库
        self.knowledge_base = KnowledgeBase(
            self.cache_dir / "knowledge_base.json"
        )

    # ========== 短期记忆操作 ==========

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """获取会话上下文"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "conversation_history": [],
                "collected_params": {},
                "current_intent": None,
                "user_goals": [],  # 用户目标历史
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话历史"""
        session = self.get_session(session_id)
        session["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        session["last_updated"] = datetime.now().isoformat()

    def get_conversation_history(
        self,
        session_id: str,
        max_turns: int = 6
    ) -> List[Dict]:
        """获取对话历史"""
        session = self.get_session(session_id)
        history = session.get("conversation_history", [])
        return history[-max_turns:] if len(history) > max_turns else history

    def get_full_context(self, session_id: str) -> Dict[str, Any]:
        """获取完整上下文，供思考引擎使用"""
        session = self.get_session(session_id)
        working_mem = self.get_working_memory(session_id)

        return {
            "conversation_history": session.get("conversation_history", []),
            "collected_params": session.get("collected_params", {}),
            "current_intent": session.get("current_intent"),
            "user_goals": session.get("user_goals", []),
            "working_state": working_mem.to_dict(),
            "accumulated_params": {
                "lesson_plan": self.accumulated_lesson_plan,
                "sports_meeting": self.accumulated_sports_meeting
            },
            "user_preferences": self.user_preferences
        }

    # ========== 工作记忆操作 ==========

    def get_working_memory(self, session_id: str) -> WorkingMemory:
        """获取工作记忆"""
        if session_id not in self.working_memories:
            self.working_memories[session_id] = WorkingMemory()
        return self.working_memories[session_id]

    def set_current_goal(self, session_id: str, goal: UserGoal):
        """设置当前目标"""
        working_mem = self.get_working_memory(session_id)
        working_mem.set_goal(goal)

        # 记录到会话历史
        session = self.get_session(session_id)
        session["user_goals"].append(asdict(goal))

    def update_task_state(self, session_id: str, state: TaskState):
        """更新任务状态"""
        working_mem = self.get_working_memory(session_id)
        working_mem.current_state = state

    # ========== 长期记忆操作 ==========

    def get_accumulated_params(self, plan_type: str) -> Dict[str, Any]:
        """获取累积参数"""
        if plan_type == "lesson_plan":
            return dict(self.accumulated_lesson_plan)
        elif plan_type == "sports_meeting":
            return dict(self.accumulated_sports_meeting)
        return {}

    def update_accumulated_params(self, plan_type: str, params: Dict[str, Any]):
        """更新累积参数"""
        if plan_type == "lesson_plan":
            for k, v in params.items():
                if v is not None and v != "":
                    self.accumulated_lesson_plan[k] = v
        elif plan_type == "sports_meeting":
            for k, v in params.items():
                if v is not None and v != "":
                    self.accumulated_sports_meeting[k] = v

    def reset_accumulated_params(self, plan_type: Optional[str] = None):
        """重置累积参数"""
        if plan_type is None or plan_type == "lesson_plan":
            self.accumulated_lesson_plan.clear()
        if plan_type is None or plan_type == "sports_meeting":
            self.accumulated_sports_meeting.clear()

    def update_user_preferences(self, key: str, value: Any):
        """更新用户偏好"""
        self.user_preferences[key] = value

    # ========== 知识库操作 ==========

    def record_experience(self, experience: Experience):
        """记录经验"""
        self.knowledge_base.add_experience(experience)

    def get_relevant_knowledge(
        self,
        scenario_type: str
    ) -> Dict[str, Any]:
        """获取相关知识"""
        return {
            "experiences": [
                asdict(e) for e in
                self.knowledge_base.get_relevant_experiences(scenario_type)
            ],
            "best_practices": self.knowledge_base.get_best_practices(scenario_type),
            "failure_patterns": self.knowledge_base.get_failure_patterns(scenario_type)
        }

    # ========== 会话管理 ==========

    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.working_memories:
            del self.working_memories[session_id]

    def reset(self, session_id: str):
        """重置会话和相关记忆"""
        self.clear_session(session_id)
        self.reset_accumulated_params()

    def save_to_file(self, session_id: str):
        """保存会话到文件"""
        if not self.cache_dir:
            return

        session = self.get_session(session_id)
        file_path = self.cache_dir / f"session_{session_id}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

    def load_from_file(self, session_id: str) -> Optional[Dict]:
        """从文件加载会话"""
        if not self.cache_dir:
            return None

        file_path = self.cache_dir / f"session_{session_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                session = json.load(f)
                self.sessions[session_id] = session
                return session
        except Exception as e:
            logger.error(f"加载会话失败: {e}")
            return None

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息摘要"""
        session = self.get_session(session_id)
        working_mem = self.get_working_memory(session_id)

        return {
            "session_id": session_id,
            "message_count": len(session.get("conversation_history", [])),
            "current_intent": session.get("current_intent"),
            "task_state": working_mem.current_state.value,
            "has_goal": working_mem.current_goal is not None,
            "strategies_count": len(working_mem.strategies),
            "accumulated_lesson_plan": self.accumulated_lesson_plan,
            "accumulated_sports_meeting": self.accumulated_sports_meeting,
        }

