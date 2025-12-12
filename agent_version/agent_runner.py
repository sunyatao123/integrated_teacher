"""
LangGraph Agent 运行器 - 使用 create_react_agent 构建真正的 ReAct Agent
参考: https://langgraph.com.cn/agents/agents.1.html
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

# 确保模块路径正确
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from tools.search_tools import ALL_TOOLS

# ============================================================
# 日志配置
# ============================================================
LOG_DIR = _CURRENT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 创建日志器
logger = logging.getLogger("Agent")
logger.setLevel(logging.DEBUG)

# 终端handler - 简要信息
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
))

# 文件handler - 详细信息
log_file = LOG_DIR / f"agent_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# 清除已有handler避免重复
logger.handlers.clear()
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 是否开启调试日志
DEBUG_MODE = os.getenv("DEBUG_AGENT", "1") == "1"

# 全局 checkpointer 用于多轮对话记忆
_checkpointer = InMemorySaver()

# 缓存 agent 实例
_agent_instance = None


# ============================================================
# 加载系统提示词
# ============================================================
def load_system_prompt() -> str:
    """从文件加载系统提示词"""
    prompt_file = _CURRENT_DIR / "prompts" / "system_prompt.txt"
    
    if prompt_file.exists():
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        logger.warning(f"[PROMPT] 提示词文件不存在: {prompt_file}")
        # 返回默认提示词
        return """你是一个专业的AI备课助手，专门为小学体育老师设计课课练方案和全员运动会方案。

你有以下工具可以使用：
1. lesson_plan_search: 课课练内部检索
2. sports_meeting_search: 运动会内部检索  
3. web_search: 互联网搜索
4. load_class_profile: 班级画像

请根据用户需求选择合适的工具，生成专业的方案。
"""


# ============================================================
# 日志辅助函数
# ============================================================
def _log_console(msg: str):
    """终端简要日志"""
    logger.info(msg)


def _log_file(msg: str):
    """文件详细日志"""
    logger.debug(msg)


def _log_separator(title: str = ""):
    """日志分隔符"""
    sep = "=" * 60
    if title:
        _log_file(f"\n{sep}\n{title}\n{sep}")
    else:
        _log_file(sep)


def _log_tool_call(tool_name: str, args: dict):
    """记录工具调用"""
    args_str = json.dumps(args, ensure_ascii=False, indent=2)
    
    # 终端简要显示
    args_brief = ", ".join(f"{k}={v}" for k, v in args.items() if v)
    _log_console(f"[TOOL] {tool_name}({args_brief})")
    
    # 文件详细记录
    _log_file(f"[TOOL_CALL] {tool_name}")
    _log_file(f"[TOOL_ARGS]\n{args_str}")


def _log_tool_result(tool_name: str, result: str):
    """记录工具结果"""
    try:
        result_json = json.loads(result)
        success = result_json.get("success", False)
        count = result_json.get("count", 0)
        source = result_json.get("source", "unknown")
        error = result_json.get("error", "")
        results = result_json.get("results", [])
        
        # 终端简要显示
        if success:
            _log_console(f"       -> OK | source={source} | count={count}")
        else:
            _log_console(f"       -> FAIL | error={error}")
        
        # 文件详细记录
        _log_file(f"[TOOL_RESULT] {tool_name}")
        _log_file(f"  success: {success}")
        _log_file(f"  source: {source}")
        _log_file(f"  count: {count}")
        if error:
            _log_file(f"  error: {error}")
        
        # 记录结果详情
        if results:
            _log_file(f"  results:")
            for i, r in enumerate(results[:5]):  # 最多记录5条
                title = r.get("title") or r.get("name") or "无标题"
                text_preview = (r.get("text") or r.get("description") or "")[:100]
                _log_file(f"    [{i+1}] {title}")
                if text_preview:
                    _log_file(f"        {text_preview}...")
    except Exception as e:
        preview = result[:500] + "..." if len(result) > 500 else result
        _log_file(f"[TOOL_RESULT] {tool_name} (raw)\n{preview}")


# ============================================================
# Agent 创建和运行
# ============================================================
def get_agent():
    """创建并返回 ReAct Agent（使用缓存）"""
    global _agent_instance
    
    if _agent_instance is not None:
        return _agent_instance
    
    _log_console("[INIT] Creating Agent...")
    _log_file("[INIT] Starting Agent initialization")
    
    model_name = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")
    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    
    _log_file(f"[INIT] Model: {model_name}")
    _log_file(f"[INIT] API: {base_url}")
    
    # 加载系统提示词
    system_prompt = load_system_prompt()
    _log_file(f"[INIT] System prompt loaded, length: {len(system_prompt)}")
    
    # 初始化模型
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("SILICONFLOW_API_KEY", ""),
        base_url=base_url,
        temperature=0.7,
    )
    
    # 使用 create_react_agent 创建 Agent
    _agent_instance = create_react_agent(
        model=model,
        tools=ALL_TOOLS,
        prompt=system_prompt,
        checkpointer=_checkpointer,
    )
    
    _log_console("[INIT] Agent ready")
    _log_file(f"[INIT] Agent initialized, tools: {[t.name for t in ALL_TOOLS]}")
    
    return _agent_instance


def run_agent_dialog(
    user_text: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    thread_id: Optional[str] = None
) -> str:
    """运行 Agent 对话（非流式）"""
    agent = get_agent()
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    _log_separator(f"Request - {datetime.now().strftime('%H:%M:%S')}")
    _log_console(f"[USER] {user_text[:50]}...")
    _log_file(f"[USER_INPUT] {user_text}")
    
    # 构建消息
    messages = []
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": user_text})
    
    # 调用 Agent
    result = agent.invoke({"messages": messages}, config=config)
    
    # 提取最后一条 AI 消息
    if result.get("messages"):
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    continue
                _log_console(f"[DONE] length={len(msg.content)}")
                _log_file(f"[OUTPUT] length: {len(msg.content)}")
                _log_file(f"[OUTPUT_CONTENT]\n{msg.content[:1000]}...")
                return msg.content
    
    return "抱歉，生成失败，请重试。"


def run_agent_stream(
    user_text: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    thread_id: Optional[str] = None
) -> Generator[str, None, None]:
    """运行 Agent 对话（流式）"""
    agent = get_agent()
    
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    _log_separator(f"Request (stream) - {datetime.now().strftime('%H:%M:%S')}")
    _log_console(f"[USER] {user_text[:50]}...")
    _log_file(f"[USER_INPUT] {user_text}")
    
    # 构建消息
    messages = []
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
    
    messages.append({"role": "user", "content": user_text})
    
    # 跟踪已输出的内容
    output_content = ""
    tool_call_count = 0
    
    try:
        for event in agent.stream({"messages": messages}, config=config, stream_mode="messages"):
            if isinstance(event, tuple) and len(event) >= 1:
                msg = event[0]
                
                # 处理 AI 消息块（流式token）
                if isinstance(msg, AIMessageChunk):
                    if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                        for tc in msg.tool_call_chunks:
                            if tc.get("name"):
                                tool_call_count += 1
                                _log_file(f"[TOOL_CHUNK] Preparing: {tc.get('name')}")
                    elif msg.content:
                        yield msg.content
                        output_content += msg.content
                
                # 处理完整的 AI 消息
                elif isinstance(msg, AIMessage):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            _log_tool_call(tc.get("name", "unknown"), tc.get("args", {}))
                    elif msg.content and msg.content not in output_content:
                        yield msg.content
                        output_content += msg.content
                
                # 处理工具结果
                elif isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "unknown")
                    _log_tool_result(tool_name, msg.content)
        
        _log_console(f"[DONE] length={len(output_content)} | tools={tool_call_count}")
        _log_file(f"[COMPLETE] Output length: {len(output_content)}, Tool calls: {tool_call_count}")
        _log_file(f"[OUTPUT_PREVIEW]\n{output_content[:500]}...")
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"[ERROR] Stream failed: {str(e)}")
        _log_file(f"[ERROR]\n{error_msg}")
        yield f"\n\n生成失败: {str(e)}"


# ============================================================
# 会话管理
# ============================================================
_sessions: Dict[str, str] = {}


def get_or_create_thread_id(session_id: str) -> str:
    """获取或创建会话的 thread_id"""
    if session_id not in _sessions:
        _sessions[session_id] = str(uuid.uuid4())
        _log_file(f"[SESSION] New session: {session_id}")
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    """重置会话"""
    if session_id in _sessions:
        del _sessions[session_id]
        _log_file(f"[SESSION] Reset: {session_id}")


# 启动时输出日志文件位置
print(f"\n[LOG] Log file: {log_file}\n")
