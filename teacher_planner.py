#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Tuple, Any
from pathlib import Path

import requests
from ai_model_optimized import OptimizedAIModel

# 配置日志
def setup_logger():
    """配置日志系统"""
    # 创建logs目录
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # 创建logger
    logger = logging.getLogger("teacher_planner")
    logger.setLevel(logging.DEBUG if os.getenv('DEBUG_AI', '1') == '1' else logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 创建文件handler（带轮转）
    log_file = log_dir / "teacher_planner.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台handler（可选，用于开发调试）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化logger
logger = setup_logger()

# 长期记忆：累积参数存储（按意图类型分开）
_accumulated_lesson_plan: Dict[str, Any] = {}
_accumulated_sports_meeting: Dict[str, Any] = {}

def reset_accumulated_params(plan_type: str = None):
    """
    重置累积参数（长期记忆）

    参数：
        plan_type: 指定重置哪个意图的记忆，None 表示全部重置
    """
    global _accumulated_lesson_plan, _accumulated_sports_meeting
    if plan_type is None or plan_type == "lesson_plan":
        _accumulated_lesson_plan.clear()
    if plan_type is None or plan_type == "sports_meeting":
        _accumulated_sports_meeting.clear()

# 提示词模板加载函数
def load_prompt_template(template_name: str) -> str:
    """从prompts文件夹加载提示词模板"""
    template_path = Path(__file__).parent / "prompts" / f"{template_name}.txt"
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"提示词模板 {template_name}.txt 未找到")
        return ""

# 加载班级配置
def load_class_profiles() -> Dict[str, Any]:
    """从prompts文件夹加载班级配置"""
    profiles_path = Path(__file__).parent / "prompts" / "class_profiles.json"
    try:
        with open(profiles_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("班级配置文件 class_profiles.json 未找到")
        return {}

# 加载系统提示词
TEACHER_SYSTEM_PROMPT = load_prompt_template("teacher_system_prompt")

def _normalize_class_name(text: str) -> str:
    """
    规范化班级名称，将中文数字转换为阿拉伯数字
    例如："三年级五班" -> "三年级5班"
    """
    # 中文数字到阿拉伯数字的映射
    cn_num_map = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
    }

    result = text
    for cn, num in cn_num_map.items():
        result = result.replace(cn, num)

    return result


def detect_class_and_fill_params(user_text: str, intent: str = "lesson_plan") -> Tuple[bool, Dict[str, Any]]:
    """
    检测用户输入是否包含班级名称，如果包含则自动填充参数

    参数：
        user_text: 用户输入文本
        intent: 意图类型（目前只支持 lesson_plan）

    返回：
        (是否检测到班级, 预填充的参数字典)

    示例：
        用户输入："一年级一班的课课练"
        返回：(True, {"grades_query": "1", "trained_weaknesses": "速度", "count_query": "", ...})
    """
    # 只有课课练意图才支持班级检测
    if intent != "lesson_plan":
        return False, {}

    # 加载班级配置
    class_profiles = load_class_profiles()

    # 如果配置文件为空，直接返回
    if not class_profiles:
        return False, {}

    # 规范化用户输入（将中文数字转换为阿拉伯数字）
    normalized_user_text = _normalize_class_name(user_text)

    # 【改进】使用更精确的匹配逻辑
    # 按班级名称长度从长到短排序，优先匹配更长的班级名称（避免"一年级一班"匹配到"一年级"）
    sorted_classes = sorted(class_profiles.items(), key=lambda x: len(x[0]), reverse=True)

    # 检测用户输入中是否包含班级名称（完全匹配）
    for class_name, class_info in sorted_classes:
        # 同时规范化班级名称，确保匹配一致性
        normalized_class_name = _normalize_class_name(class_name)
        if normalized_class_name in normalized_user_text:
            # 【新增】验证匹配的有效性：确保不是部分匹配
            # 例如："一年级一班" 不应该匹配 "一年级三班"
            # 通过检查班级名称前后的字符来验证
            idx = normalized_user_text.find(normalized_class_name)
            if idx != -1:
                # 检查前后字符，确保是完整的班级名称
                before_char = normalized_user_text[idx - 1] if idx > 0 else " "
                after_char = normalized_user_text[idx + len(normalized_class_name)] if idx + len(normalized_class_name) < len(normalized_user_text) else " "

                # 【修复】改进匹配逻辑：
                # 1. 如果班级名称本身包含"班"（如"一年级一班"），后面不应该再有数字或"班"
                # 2. 如果班级名称不包含"班"（如"kkk"），后面可以有"班"字（如"kkk班级"）
                # 3. 前面不应该有数字（避免"1一年级一班"这种情况）
                is_valid_match = True

                # 前面不应该有数字
                if before_char.isdigit():
                    is_valid_match = False

                # 如果班级名称包含"班"，后面不应该再有数字或"班"
                if "班" in normalized_class_name and (after_char.isdigit() or after_char == "班"):
                    is_valid_match = False

                # 如果班级名称不包含"班"，后面不应该有数字（但可以有"班"）
                if "班" not in normalized_class_name and after_char.isdigit():
                    is_valid_match = False

                if is_valid_match:
                    # 找到匹配的班级，返回预填充的参数
                    params = {
                        "semantic_query": class_info.get("semantic_query", ""),
                        "count_query": class_info.get("count_query", ""),
                        "grades_query": class_info.get("grades_query", ""),
                        "trained_weaknesses": class_info.get("trained_weaknesses", ""),
                        "top_k": 10,
                        "detected_class_name": class_name  # 【新增】记录检测到的班级名称
                    }
                    logger.info(f"[班级检测] 识别到班级: {class_name}")
                    logger.info(f"[班级检测] 自动填充参数: {json.dumps(params, ensure_ascii=False)}")
                    logger.info("================================")
                    return True, params

    # 没有检测到班级
    logger.info("[班级检测] 未识别到配置文件中的班级")
    logger.info("================================")
    return False, {}

def detect_intent_llm(user_text: str, conversation_history: List[Dict[str, str]] = None, timeout: float = 15.0) -> str:
    """
    使用大模型进行意图识别，判断用户是想进行：
    - sports_meeting: 全员运动会方案设计
    - lesson_plan: 课课练方案设计
    - chat: 闲聊或其他
    
    参数：
        user_text: 当前用户输入
        conversation_history: 对话历史记录
    返回：
        "sports_meeting" | "lesson_plan" | "chat"
    """
    model = OptimizedAIModel()
    system = load_prompt_template("intent_recognition")
    
    # 构建历史对话上下文
    history_text = ""
    if conversation_history:
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        history_lines = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_lines.append(f"用户：{content}")
            elif role == "assistant":
                history_lines.append(f"助手：{content}")
        if history_lines:
            history_text = "\n".join(history_lines)
    
    user = f"""
对话历史（最近6轮）：
{history_text if history_text else "（无历史记录）"}

当前用户输入：{user_text}

请判断用户的意图，只输出JSON。
""".strip()
    
    # 记录模型调用message
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    logger.info(f"[MODEL_CALL] 意图识别 - message:{json.dumps(messages, ensure_ascii=False, indent=2)}")
    start_time = time.time()
    
    resp = model.client.chat.completions.create(
        model=model.model,
        messages=messages,
        max_tokens=100,
        temperature=0.1,
    )
    
    elapsed_time = time.time() - start_time
    content = resp.choices[0].message.content.strip()
    
    # 记录模型调用Response
    logger.info(f"[MODEL_CALL] 意图识别 - Response: {repr(content)}")
    logger.info(f"  响应时间: {elapsed_time:.3f}秒")
    logger.info("================================")
    # JSON截取
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            intent = parsed.get("intent", "chat")
            if intent in ("sports_meeting", "lesson_plan", "external_search", "chat"):
                return intent
        except Exception:
            pass
    return "chat"


def collect_entities_llm(
    user_text: str, conversation_history: List[Dict[str, str]] = None, plan_type: str = "", timeout: float = 15.0
) -> Tuple[Dict[str, Any], List[str]]:
    """
    使用大模型进行实体抽取，从用户输入和对话历史中提取参数

    参数：
        user_text: 用户输入文本
        conversation_history: 对话历史
        plan_type: 意图类型 ("sports_meeting" | "lesson_plan" | "chat")
        timeout: 超时时间

    返回：(提取的参数字典, 缺失的字段列表)
    """
    global _accumulated_lesson_plan, _accumulated_sports_meeting

    # 根据意图类型选择累积变量
    if plan_type == "lesson_plan":
        accumulated = _accumulated_lesson_plan
    elif plan_type == "sports_meeting":
        accumulated = _accumulated_sports_meeting
    else:
        accumulated = {}
    
    # 【新增】检测新会话：如果对话历史为空，清空累积变量
    if not conversation_history or len(conversation_history) == 0:
        accumulated.clear()
        
    # 【新增】优先检测班级场景，如果检测到班级，直接返回预填充的参数
    # 注意：这里假设是lesson_plan意图，因为只有课课练才支持班级检测
    is_class, class_params = detect_class_and_fill_params(user_text, intent=plan_type)
    if is_class:
        # 将班级参数合并到累积变量（非空才覆盖）
        for key, value in class_params.items():
            if value is not None and value != "":
                accumulated[key] = value
        # logger.info(f"[班级检测] 累积变量：{accumulated}")
        # 不直接返回，继续进行 LLM 实体抽取以提取用户输入中的其他信息

    model = OptimizedAIModel()
    system = load_prompt_template("param_extraction_system")
    
    # 构建历史对话上下文（至少3轮，最多6轮）
    history_text = ""
    if conversation_history:
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        history_lines = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_lines.append(f"用户：{content}")
            elif role == "assistant":
                history_lines.append(f"助手：{content}")
        if history_lines:
            history_text = "\n".join(history_lines)
    
    # 加载参数提取用户提示词模板
    user_template = load_prompt_template("param_extraction_user")
    user = user_template.format(
        history_text=history_text if history_text else "（无历史记录）",
        user_text=user_text,
    )

    # 记录模型调用message
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    logger.info(f"[MODEL_CALL] 参数提取 - message:{json.dumps(messages, ensure_ascii=False, indent=2)}")
    start_time = time.time()

    resp = model.client.chat.completions.create(
        model=model.model,
        messages=messages,
        max_tokens=400,
        temperature=0.2,
        response_format={'type': 'json_object'},  #deepseek的json输出格式
    )
    
    elapsed_time = time.time() - start_time
    content = resp.choices[0].message.content.strip()

    # 记录模型调用Response
    logger.info(f"[MODEL_CALL] 参数提取 - Response:{repr(content)}")
    logger.info(f"  响应时间: {elapsed_time:.3f}秒")
    logger.info("================================")
    # JSON截取
    start = content.find("{")
    end = content.rfind("}")
    parsed: Dict[str, Any] = {}
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
        except Exception as e:
            logger.error(f"[PARAM_EXTRACTION] JSON解析失败: {e}")
            parsed = {}

    # 将新抽取的参数合并到累积变量（非空才覆盖）
    for key, value in parsed.items():
        if value is not None and value != "":
            accumulated[key] = value

    # 根据意图类型决定提取哪些字段（基于累积变量判断缺失）
    if plan_type == "sports_meeting":
        # 全员运动会：提取操场条件、年级、人数
        missing: List[str] = []
        if not accumulated.get("semantic_query"):
            missing.append("semantic_query")
    elif plan_type == "lesson_plan":
        # 课课练：需要年级 + (project_name 或 弱项)，满足其一即可
        missing: List[str] = []
        has_grades = bool(accumulated.get("grades_query"))
        has_weaknesses = bool(accumulated.get("trained_weaknesses"))
        has_project_name = bool(accumulated.get("project_name"))
        
        # 必须有年级
        if not has_grades:
            missing.append("grades_query")
        
        # project_name 或 弱项 满足其一即可
        if not has_project_name and not has_weaknesses:
            missing.append("trained_weaknesses")
    else:
        # 闲聊模式：不需要提取业务字段，不检查缺失
        missing: List[str] = []

    logger.info(f"[PARAM_EXTRACTION] 长期记忆信息收集: {accumulated}")
    logger.info(f"[PARAM_EXTRACTION] 缺失字段: {missing}")
    logger.info("================================")

    # 返回副本，避免外部修改影响累积变量
    return dict(accumulated), missing


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 8.0) -> List[Dict[str, Any]]:
    # 记录请求信息
    if os.getenv('DEBUG_AI','1')=='1':
        logger.info(f"[TEACHER] 🚀 检索接口请求")
        logger.info(f"   URL: {url}")
        logger.info(f"   检索参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    resp = requests.post(url, json=payload, timeout=timeout)


    if resp.status_code != 200:
        # 记录详细的错误信息
        try:
            error_detail = resp.text
            if os.getenv('DEBUG_AI','1')=='1':
                logger.error(f"[TEACHER] 检索接口错误详情: status_code={resp.status_code}, response={error_detail}")
        except:
            pass
    resp.raise_for_status()
    data = resp.json()

    # 记录返回结果数量
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        if os.getenv('DEBUG_AI','1')=='1':
            logger.info(f"[TEACHER] 📊 检索返回 {len(results)} 条结果")
        return results
    if isinstance(data, list):
        if os.getenv('DEBUG_AI','1')=='1':
            logger.info(f"[TEACHER] 📊 检索返回 {len(data)} 条结果")
        return data

    if os.getenv('DEBUG_AI','1')=='1':
        logger.warning(f"[TEACHER] ⚠️ 检索返回空结果")
    return []


def call_lesson_plan_search(base_url, payload, timeout=8.0):
    url = base_url + "/extended-search/hybrid"
    return _post_json(url, payload, timeout)


def call_sports_meeting_search(base_url, payload, timeout=8.0):
    url = base_url + "/search/hybrid"
    return _post_json(url, payload, timeout)


# ==================== 互联网搜索相关函数 ====================

def build_web_search_query(user_text: str, params: Dict[str, Any], plan_type: str = "lesson_plan") -> str:
    """
    构建互联网搜索查询词
    
    参数:
        user_text: 用户原始输入
        params: 提取的参数
        plan_type: 方案类型
    
    返回:
        str: 搜索查询词
    """
    if plan_type == "lesson_plan":
        # 课课练：构建包含年级、语义查询和薄弱项的查询
        grades_query = params.get("grades_query", "")
        trained_weaknesses = params.get("trained_weaknesses", "")
        semantic_query = params.get("semantic_query", "")  # 用户真正要搜索的内容（如篮球、跳绳等）
        project_name = params.get("project_name", "")  # 项目名称
        
        query_parts = []
        if grades_query:
            # 将数字年级转换为中文
            grade_map = {"1": "一年级", "2": "二年级", "3": "三年级", 
                        "4": "四年级", "5": "五年级", "6": "六年级"}
            grade_cn = grade_map.get(grades_query, f"{grades_query}年级")
            query_parts.append(f"小学{grade_cn}")
        
        query_parts.append("体育")
        
        # 【关键】加入用户真正要搜索的内容（语义查询或项目名称）
        if semantic_query and semantic_query.strip():
            query_parts.append(semantic_query.strip())
        elif project_name and project_name.strip():
            query_parts.append(project_name.strip())
        
        if trained_weaknesses and trained_weaknesses.strip() != "无要求":
            query_parts.append(trained_weaknesses)
        
        query_parts.append("训练方法 教案")
        
        query = " ".join(query_parts)
    else:
        # 运动会：使用用户输入和语义查询
        semantic_query = params.get("semantic_query", "")
        query = f"{semantic_query} {user_text}".strip() if semantic_query else user_text
    
    if os.getenv('DEBUG_AI', '1') == '1':
        logger.info(f"[WEB_SEARCH] 构建搜索查询词: {query}")
    
    return query


def call_tavily_search(query: str, min_results: int = 5, timeout: float = 15.0) -> List[Dict[str, Any]]:
    """
    调用Tavily API进行互联网搜索
    
    参数:
        query: 搜索查询词
        min_results: 需要的最少结果数量（会请求该数量的结果）
        timeout: 超时时间（秒），默认15秒以确保有足够时间获取结果
    
    返回:
        List[Dict]: 搜索结果列表，格式与内部结果保持一致
    
    注意:
        Tavily API 的 max_results 参数限制为20（基础版），
        如果需要更多结果，建议使用高级搜索深度或多次查询
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    tavily_api_url = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")
    
    if not tavily_api_key:
        logger.warning("[WEB_SEARCH] Tavily API Key未配置，跳过互联网搜索")
        return []
    
    # Tavily API 基础版 max_results 上限为20，如果需要更多则使用advanced深度
    search_depth = "advanced" if min_results > 10 else "basic"
    # 请求数量不能超过API限制（基础版20，高级版可能更多）
    request_count = min(min_results, 50)  # 设置上限为50
    
    try:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False,
            "max_results": request_count
        }
        
        # 打印请求参数（隐藏API key）
        if os.getenv('DEBUG_AI', '1') == '1':
            logger.info(f"[WEB_SEARCH] ====== Tavily API 请求 ======")
            logger.info(f"[WEB_SEARCH] 请求参数:")
            logger.info(f"   - query: {query}")
            logger.info(f"   - search_depth: {search_depth}")
            logger.info(f"   - include_answer: True")
            logger.info(f"   - include_raw_content: False")
            logger.info(f"   - max_results: {request_count}")
            logger.info(f"   - timeout: {timeout}秒")
        
        resp = requests.post(
            tavily_api_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        
        resp.raise_for_status()
        data = resp.json()
        
        # 打印API原始返回数据
        if os.getenv('DEBUG_AI', '1') == '1':
            logger.info(f"[WEB_SEARCH] ====== Tavily API 响应 ======")
            logger.info(f"[WEB_SEARCH] HTTP状态码: {resp.status_code}")
            # logger.info(f"[WEB_SEARCH] 原始返回数据:")
            # logger.info(json.dumps(data, ensure_ascii=False, indent=2))
        
        # 转换Tavily结果格式为内部结果格式
        results = []
        tavily_results = data.get("results", [])
        answer = data.get("answer", "")
        
        # 如果有answer，优先使用
        if answer:
            results.append({
                "text": answer,
                "title": f"关于'{query}'的搜索结果",
                "source": "tavily_answer",
                "url": "",
                "_source": "web"
            })
        
        # 添加其他搜索结果
        for item in tavily_results[:request_count]:
            content = item.get("content", "")
            title = item.get("title", "")
            url = item.get("url", "")
            
            if content:
                results.append({
                    "text": content,
                    "title": title,
                    "source": "tavily",
                    "url": url,
                    "_source": "web"
                })
        
        # 打印转换后的结果
        if os.getenv('DEBUG_AI', '1') == '1':
            logger.info(f"[WEB_SEARCH] ====== 转换后的结果 ======")
            logger.info(f"[WEB_SEARCH] 结果数量: {len(results)} 条")
            for i, r in enumerate(results):
                logger.info(f"[WEB_SEARCH] 结果 {i+1}:")
                logger.info(f"   - 标题: {r.get('title', '')[:100]}")
                logger.info(f"   - 来源: {r.get('source', '')}")
                logger.info(f"   - URL: {r.get('url', '')}")
                text_preview = r.get('text', '')[:200] + '...' if len(r.get('text', '')) > 200 else r.get('text', '')
                logger.info(f"   - 内容预览: {text_preview}")
            logger.info(f"[WEB_SEARCH] ====== Tavily搜索完成 ======")
        
        return results
        
    except requests.exceptions.Timeout:
        logger.error(f"[WEB_SEARCH] Tavily API调用超时（{timeout}秒）")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"[WEB_SEARCH] Tavily API调用失败: {e}")
        return []
    except Exception as e:
        logger.error(f"[WEB_SEARCH] Tavily搜索异常: {e}")
        return []


def analyze_external_search_need(
    user_text: str, 
    internal_results: List[Dict[str, Any]] = None,
    params: Dict[str, Any] = None,
    conversation_history: List[Dict[str, str]] = None,
    timeout: float = 10.0
    ) -> Tuple[bool, List[int]]:
    """
    使用LLM分析用户输入和内部检索结果，判断是否需要外部检索补充
    
    参数:
        user_text: 用户原始输入
        internal_results: 内部检索结果列表
        params: 已提取的参数
        conversation_history: 对话历史记录
        timeout: 超时时间
    
    返回:
        Tuple[bool, List[int]]: (是否需要外部检索, 相关结果索引列表)
    """
    model = OptimizedAIModel()
    
    internal_results = internal_results or []
    
    # 构建包含历史用户输入的 user_text
    user_text_with_history = user_text
    if conversation_history:
        # 提取历史对话中的用户输入（最近6轮）
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        history_user_inputs = []
        for msg in recent_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    history_user_inputs.append(content)
        # 如果有历史用户输入，合并到当前用户输入
        if history_user_inputs:
            # 将历史用户输入和当前用户输入合并，用换行分隔
            user_text_with_history = "\n".join(history_user_inputs + [user_text])
    
    # 只提取 description 字段
    internal_descriptions = ""
    if internal_results:
        descriptions = []
        for idx, r in enumerate(internal_results[:10]):  # 最多10条
            desc = r.get("description") or r.get("desc") or ""
            if desc:
                descriptions.append(f"索引{idx}：{desc.strip()}")
        internal_descriptions = "\n".join(descriptions) if descriptions else "（无内部检索结果）"
    else:
        internal_descriptions = "（无内部检索结果）"
    
    # 从文件加载提示词模板
    user_template = load_prompt_template("analyze_external_search")
    
    # 转义 internal_descriptions 中的大括号，避免与 format 占位符冲突
    # 将 { 替换为 {{，} 替换为 }}
    escaped_descriptions = internal_descriptions.replace("{", "{{").replace("}", "}}")
    
    # 使用模板格式化用户提示词
    user_prompt = user_template.format(
        user_text=user_text_with_history,
        internal_descriptions=escaped_descriptions
    )
    # logger.info(f"判断是否需要联网: {user_prompt}")
    system_prompt = "你是一个智能分析助手，负责判断是否需要从互联网补充检索结果。只输出JSON对象，不要任何解释。"

    try:
        # 记录模型调用message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        logger.info(f"[MODEL_CALL] 外部搜索分析 - message:{json.dumps(messages, ensure_ascii=False, indent=2)}")
        start_time = time.time()
        
        resp = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=300,
            temperature=0.1,
            response_format={'type': 'json_object'},
        )
        
        elapsed_time = time.time() - start_time
        content = resp.choices[0].message.content.strip()
        
        # 记录模型调用Response
        logger.info(f"[MODEL_CALL] 外部搜索分析 - Response: {repr(content)}")
        logger.info(f"  响应时间: {elapsed_time:.3f}秒 ")
        logger.info("================================")
        
        # JSON解析
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(content[start : end + 1])
            except json.JSONDecodeError as e:
                logger.error(f"[EXTERNAL_SEARCH_ANALYSIS] JSON解析失败: {e}")
                logger.error(f"[EXTERNAL_SEARCH_ANALYSIS] 原始内容: {repr(content)}")
                raise
            need_external = parsed.get("need_external", False)
            relevant_indices = parsed.get("relevant_indices", [])  # 相关结果的索引列表
            
            # 验证索引有效性
            if not isinstance(relevant_indices, list):
                relevant_indices = []
            # 过滤掉无效索引
            internal_results_count = len(internal_results)
            relevant_indices = [i for i in relevant_indices if isinstance(i, int) and 0 <= i < internal_results_count]
            
            
            return need_external, relevant_indices
            
    except Exception as e:
        logger.error(f"[EXTERNAL_SEARCH_ANALYSIS] 分析失败: {e}")
    
    # 默认策略：如果内部结果为空，需要外部检索
    if len(internal_results) == 0:
        return True, []
    
    return False, []


def build_external_search_messages(
    user_text: str,
    web_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None,
) -> tuple[List[Dict[str, str]], str]:
    """
    构建 external_search 意图的消息，用于生成回答
    
    参数:
        user_text: 用户原始输入
        web_results: 互联网检索结果
        conversation_history: 对话历史
    
    返回:
        (messages列表, results_text)，可直接用于chat.completions.create
    """
    # 加载外部检索提示词模板
    template = load_prompt_template("plan_generation_external_search")
    
    # 格式化互联网结果
    web_results_text = format_results_text(web_results, source="web", top_k=10)
    
    if not web_results_text:
        web_results_text = "未找到相关互联网资源，请根据你的专业知识回答用户问题。"
    
    # 构建用于前端展示的 results_text（只包含外部资源）
    results_text = ""
    if web_results:
        results_parts = []
        web_text = format_results_text(web_results, source="web", top_k=len(web_results))
        if web_text:
            results_parts.append(f"【互联网资源检索结果】\n{web_text}")
        results_text = "\n\n".join(results_parts) if results_parts else ""
    
    # 构建用户提示词
    user_prompt = template.format(
        user_text=user_text,
        web_results_text=web_results_text
    )
    
    # 构建消息列表
    messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]
    
    # 添加对话历史
    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append(msg)
    
    messages.append({"role": "user", "content": user_prompt})
    
    return messages, results_text


def generate_external_search_response(
    user_text: str,
    web_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None,
) -> str:
    """
    非流式生成外部检索意图的回答
    
    参数:
        user_text: 用户原始输入
        web_results: 互联网检索结果
        conversation_history: 对话历史
    
    返回:
        生成的回答文本
    """
    model = OptimizedAIModel()
    messages, _ = build_external_search_messages(user_text, web_results, conversation_history)
    
    try:
        # 记录模型调用message
        logger.info(f"[MODEL_CALL] 外部搜索生成 - message:")
        logger.info(f"  模型: {model.model}")
        logger.info(f"  Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        logger.info(f"  参数: max_tokens=3000, temperature=0.7, top_p=0.9")
        start_time = time.time()
        
        resp = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=3000,
            temperature=0.7,
            top_p=0.9,
        )
        
        elapsed_time = time.time() - start_time
        content = resp.choices[0].message.content.strip()
        
        # 记录模型调用Response
        logger.info(f"[MODEL_CALL] 外部搜索生成 - Response:")
        logger.info(f"  响应时间: {elapsed_time:.3f}秒 ({elapsed_time*1000:.1f}毫秒)")
        logger.info(f"  响应内容长度: {len(content)} 字符")
        logger.info(f"  响应内容: {repr(content)}")
        
        return content
    except Exception as e:
        logger.error(f"[EXTERNAL_SEARCH] 生成失败: {e}")
        return f"生成失败: {str(e)}"


def generate_external_search_response_stream(
    user_text: str,
    web_results: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None,
):
    """
    流式生成外部检索意图的回答
    
    参数:
        user_text: 用户原始输入
        web_results: 互联网检索结果
        conversation_history: 对话历史
    
    返回:
        生成器，逐块返回生成的文本
    """
    model = OptimizedAIModel()
    messages, results_text = build_external_search_messages(user_text, web_results, conversation_history)
    
    try:
        # 记录模型调用message
        logger.info(f"[MODEL_CALL] 外部搜索生成(流式) - message:")
        logger.info(f"  模型: {model.model}")
        logger.info(f"  Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        logger.info(f"  参数: max_tokens=3000, temperature=0.7, top_p=0.9, stream=True")
        start_time = time.time()
        full_response = ""
        
        stream = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=3000,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        
        ttfb_recorded = False
        for event in stream:
            try:
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if not ttfb_recorded:
                        ttfb_time = time.time() - start_time
                        ttfb_recorded = True
                    full_response += content
                    yield content
            except Exception:
                chunk = None
                try:
                    chunk = event["choices"][0]["delta"].get("content")
                except Exception:
                    pass
                if chunk:
                    if not ttfb_recorded:
                        ttfb_time = time.time() - start_time
                        ttfb_recorded = True
                    full_response += chunk
                    yield chunk
        
        # 记录模型调用Response
        total_time = time.time() - start_time
        logger.info(f"[MODEL_CALL] 外部搜索生成(流式) - Response:{repr(full_response)}")
        if ttfb_recorded:
            logger.info(f"  首token: {ttfb_time:.3f}秒 ")
        logger.info(f"  总响应时间: {total_time:.3f}秒")
        logger.info("================================")
        
        # 流式输出完成后，追加 results_text
        if results_text:
            yield f"\n\n---RETRIEVAL_RESULTS_START---\n\n{results_text}\n\n---RETRIEVAL_RESULTS_END---\n\n"
    except Exception as e:
        logger.error(f"[EXTERNAL_SEARCH] 流式生成失败: {e}")
        yield f"生成失败: {str(e)}"


def format_results_text(results: List[Dict[str, Any]], source: str = "internal", top_k: int = 5) -> str:
    """
    格式化检索结果为文本，用于prompt
    
    参数:
        results: 检索结果列表
        source: 结果来源（"internal"或"web"）
        top_k: 取前k条结果
    
    返回:
        str: 格式化后的文本
    """
    if not results:
        return ""
    
    texts: List[str] = []
    source_label = "内部资源" if source == "internal" else "互联网资源"
    is_web = source == "web"
    
    for r in results[:top_k]:
        t = r.get("text")
        url = r.get("url", "") if is_web else ""  # 只有互联网资源才显示URL
        
        if t:
            content = f"[{source_label}] {str(t).strip()}"
            # 为互联网资源添加来源URL
            if is_web and url:
                content += f"\n  📎 来源链接: {url}"
            texts.append(content)
            continue
        # 回退：拼一个简要描述
        title = r.get("title") or r.get("name") or ""
        desc = r.get("description") or r.get("desc") or ""
        media = r.get("image") or r.get("cover") or r.get("thumbnail") or r.get("media_url") or r.get("img") or ""
        fallback = "\n".join(x for x in [title, desc, media] if x) or ""
        if fallback:
            content = f"[{source_label}] {fallback}"
            if is_web and url:
                content += f"\n  📎 来源链接: {url}"
            texts.append(content)
    
    return "\n\n".join(texts) if texts else ""


def build_plan_messages(
    results: List[Dict[str, Any]],
    params: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    user_text: str,
    missing:List[str] = None,
    need_guidance: bool = False,
    web_results: List[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, str]], str]:
    """
    构造用于生成备课方案的messages，供流式与非流式复用。

    参数：
        results: 检索结果列表（如果need_guidance=True，可以为空列表）
        params: 实体抽取的参数
        user_text: 用户原始输入
        need_guidance: 如果为True，生成引导语而不是方案

    返回：(messages列表, results_text)，可直接用于chat.completions.create
    """
    model = OptimizedAIModel()

    # 如果需要引导，生成引导提示
    if need_guidance:
        
        missing_info = []
        missing = missing or []
        if "semantic_query" in missing:
            missing_info.append("学生人数、年级、操场大小、跑道数量、操场条件") #运动会

        if "grades_query" in missing and "trained_weaknesses" in missing:   #课课练
            missing_info.append("年级和训练需求、训练弱项（如：想练什么、训练什么薄弱项、具体动作名称等）")
        elif "grades_query" in missing:
            missing_info.append("年级信息")
        elif "trained_weaknesses" in missing:
            missing_info.append("训练需求、训练弱项（如：想练什么、训练什么薄弱项、具体动作名称等）")

        missing_str = "、".join(missing_info) if missing_info else "无"
        # 加载引导语模板
        guidance_template = load_prompt_template("guidance_prompt")
        # 格式化引导语提示词
        user_prompt = guidance_template.format(
            user_text=user_text,
            collected_info=params,
            plan_type=params.get("plan_type") or "未确定",
            missing_info=missing_str
        )
        return [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ], ""

    # 正常生成方案
    # 汇总检索结果，控制上下文长度
    top_k = int(params.get("top_k") or 5)
    
    # 计算内部和外部检索的实际数量
    internal_count = len(results) if results else 0
    web_count = len(web_results) if web_results else 0
    total_count = internal_count + web_count
    
    # 分别处理内部和互联网结果（不限制top_k，使用全部结果）
    internal_text = format_results_text(results, source="internal", top_k=internal_count if internal_count > 0 else top_k)
    web_text = format_results_text(web_results or [], source="web", top_k=web_count if web_count > 0 else top_k)
    
    # 组合结果文本
    results_parts = []
    if internal_text:
        results_parts.append(f"【内部资源检索结果】\n{internal_text}")
    if web_text:
        results_parts.append(f"【互联网资源检索结果】\n{web_text}")
    
    results_text = "\n\n".join(results_parts) if results_parts else "无检索结果text，需由你结合参数生成通用方案。"

    # logger.info(f"[传入模型的results_text] results_text: {results_text}")
    # 使用实体抽取结果本身（不再使用默认值兜底，仅对 top_k 兜底）
    meta = {
        "semantic_query": params.get("semantic_query"),
        "count_query": str(params.get("count_query")),
        "grades_query": str(params.get("grades_query")),
        "trained_weaknesses": params.get("trained_weaknesses"),
        "top_k": int(params.get("top_k") or 10),
    }

    # 根据意图类型生成不同的提示词
    plan_type = params.get("plan_type")
    is_sports_meeting = plan_type == "sports_meeting"
    is_lesson_plan = plan_type == "lesson_plan"
    is_chat = plan_type == "chat"

    if is_sports_meeting:
        # 全员运动会方案生成提示词
        template = load_prompt_template("plan_generation_sports_meeting")
        user_prompt = template.format(
            user_text=user_text,
            conversation_history=conversation_history,
            meta=json.dumps(meta, ensure_ascii=False, indent=2),
            results_text=results_text,
            grades_query=meta.get("grades_query") or "根据用户输入确定",
            count_query=meta.get("count_query") or "根据用户输入确定",
            semantic_query=meta.get("semantic_query") or "标准操场"
        )
    elif is_lesson_plan:
        # 课课练方案生成提示词
        # 生成班级分析文本
        class_analysis_text = ""
        grades_query = params.get("grades_query")
        detected_class_name = params.get("detected_class_name")  # 【新增】获取检测到的班级名称

        # 【修复】优先使用检测到的班级名称进行精确匹配
        if detected_class_name:
            class_profiles = load_class_profiles()
            if detected_class_name in class_profiles:
                profile = class_profiles[detected_class_name]
                weakness_details = profile.get("weakness_details", {})
                student_groups = profile.get("student_groups", {})

                if weakness_details:
                    class_analysis_text = f"   - 班级：{detected_class_name}\n   - 班级薄弱项描述：\n"
                    for weakness, detail in weakness_details.items():
                        class_analysis_text += f"     * {weakness}：{detail}\n"
                    class_analysis_text += "\n"

                # 新增：添加学生分组信息
                if student_groups:
                    class_analysis_text += f"   - {detected_class_name}学生分组情况：\n"
                    for group_key, group_info in student_groups.items():
                        count = group_info.get("count", 0)
                        weakness_items = group_info.get("weakness_items", [])
                        student_details = group_info.get("student_details", [])

                        # 生成分组描述
                        class_analysis_text += f"     * {group_key}薄弱组：{count}人\n"

                        # 添加薄弱项目列表
                        if weakness_items:
                            class_analysis_text += f"       薄弱项目：{', '.join(weakness_items)}\n"

                        # 添加学生名单（包含序号和学号）
                        if student_details:
                            class_analysis_text += f"       学生名单：\n"
                            for student in student_details:
                                student_num = student.get("序号", "")
                                student_id = student.get("学生编号", "")
                                student_name = student.get("姓名", "")
                                gender = student.get("性别", "")

                                # 构建学生信息字符串
                                if student_name:
                                    student_info = f"{student_name}"
                                else:
                                    student_info = f"学生{student_num}"

                                class_analysis_text += f"         • {student_info} [{student_id}]\n"

                        class_analysis_text += "\n"

                    class_analysis_text += "   - **重要**：请在方案开头展示上述学生分组情况，并根据分组为不同薄弱项的学生推荐不同的练习！\n"

        # 如果没有找到班级配置，生成提示信息
        # if not class_analysis_text and grades_query:
        #     class_analysis_text = f"   - 由于配置中没有该班级的详细信息，本方案将基于{grades_query}年级的一般特点提供通用的练习推荐。\n   - **重要**：请在方案开头展示这个提示信息！\n"

        template = load_prompt_template("plan_generation_lesson_plan")
        user_prompt = template.format(
            user_text=user_text,
            conversation_history=conversation_history,
            meta=json.dumps(meta, ensure_ascii=False, indent=2),
            results_text=results_text,
            class_analysis_text=class_analysis_text,
            internal_count=internal_count,
            web_count=web_count,
            total_count=total_count
        )
    elif plan_type == "chat":
        # 闲聊：仅返回系统提示与原始输入
        messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append(msg)
        messages.append({"role": "user", "content": user_text})
        return messages, ""
    else:
        # 未知意图，返回默认消息
        return [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ], ""

    # 返回消息列表
    messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]
    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append(msg)
    messages.append({"role": "user", "content": user_prompt})
    return messages, results_text


def generate_plan_stream(
    results: List[Dict[str, Any]],
    params: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    user_text: str,
    missing:List[str] = None,
    need_guidance: bool = False,
    web_results: List[Dict[str, Any]] = None,
):
    """
    流式生成备课方案

    参数：
        results: 内部检索结果列表
        params: 实体抽取的参数
        conversation_history: 对话历史
        user_text: 用户原始输入
        missing: 缺失字段列表
        need_guidance: 如果为True，生成引导语而不是方案
        web_results: 互联网搜索结果列表（可选）

    返回：生成器，逐块返回生成的文本
    """
    model = OptimizedAIModel()
    missing = missing or []
    messages, results_text = build_plan_messages(results, params, conversation_history, user_text, missing, need_guidance, web_results=web_results)
    # logger.info(f"[TEACHER] 流式生成请求，messages={messages}")
    
    # 记录模型调用message
    logger.info(f"[MODEL_CALL] 方案生成/引导语生成(流式) - message: {json.dumps(messages, ensure_ascii=False, indent=2)}")
    start_time = time.time()
    full_response = ""

    try:
        stream = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        chunk_count = 0
        ttfb_recorded = False
        for event in stream:
            try:
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if not ttfb_recorded:
                        ttfb_time = time.time() - start_time
                        ttfb_recorded = True
                    chunk_count += 1
                    full_response += content
                    yield content
            except Exception:
                chunk = None
                try:
                    chunk = event["choices"][0]["delta"].get("content")
                except Exception:
                    pass
                if chunk:
                    if not ttfb_recorded:
                        ttfb_time = time.time() - start_time
                        ttfb_recorded = True
                    chunk_count += 1
                    full_response += chunk
                    yield chunk
        
        # 记录模型调用Response
        total_time = time.time() - start_time
        logger.info(f"[MODEL_CALL] 方案生成/引导语生成(流式) - Response: {repr(full_response)}")
        if ttfb_recorded:
            logger.info(f"  首token: {ttfb_time:.3f}秒 ")
        logger.info(f"  总响应时间: {total_time:.3f}秒 ")
        logger.info("================================")
        
        # 流式输出完成后，追加 results_text（仅在正常生成方案时）
        if results_text and not need_guidance:
            yield f"\n\n---RETRIEVAL_RESULTS_START---\n\n{results_text}\n\n---RETRIEVAL_RESULTS_END---\n\n"
    except Exception as e:
        if os.getenv('DEBUG_AI','1')=='1':
            logger.error(f"[TEACHER] 流式生成失败: {e}")
        yield f"生成失败: {str(e)}"


def generate_plan(
    results: List[Dict[str, Any]],
    params: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    user_text: str,
    need_guidance: bool = False,
    web_results: List[Dict[str, Any]] = None,
) -> str:
    """
    非流式生成备课方案

    参数：
        results: 内部检索结果列表
        params: 实体抽取的参数
        conversation_history: 对话历史
        user_text: 用户原始输入
        need_guidance: 如果为True，生成引导语而不是方案
        web_results: 互联网搜索结果列表（可选）

    返回：生成的文本
    """
    model = OptimizedAIModel()
    messages, _ = build_plan_messages(results, params, conversation_history, user_text, need_guidance=need_guidance, web_results=web_results)

    try:
        # 记录模型调用message
        logger.info(f"[MODEL_CALL] 方案生成 - message:")
        logger.info(f"  模型: {model.model}")
        logger.info(f"  Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        logger.info(f"  参数: max_tokens=3000, temperature=0.7, top_p=0.9")
        start_time = time.time()
        
        resp = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=3000,
            temperature=0.7,
            top_p=0.9,
        )
        
        elapsed_time = time.time() - start_time
        content = resp.choices[0].message.content.strip()
        
        # 记录模型调用Response
        logger.info(f"[MODEL_CALL] 方案生成 - Response:")
        logger.info(f"  响应时间: {elapsed_time:.3f}秒 ({elapsed_time*1000:.1f}毫秒)")
        logger.info(f"  响应内容长度: {len(content)} 字符")
        logger.info(f"  响应内容: {repr(content)}")
        
        return content
    except Exception as e:
        logger.error(f"生成失败: {e}")
        return f"生成失败: {str(e)}"

