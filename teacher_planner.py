#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import logging
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
                    return True, params

    # 没有检测到班级
    logger.info("[班级检测] 未识别到配置文件中的班级")
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
    
    resp = model.client.chat.completions.create(
        model=model.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=100,
        temperature=0.1,
    )
    content = resp.choices[0].message.content.strip()
    
    # JSON截取
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            intent = parsed.get("intent", "chat")
            if intent in ("sports_meeting", "lesson_plan", "chat"):
                return intent
        except Exception:
            pass
    return "chat"


def collect_entities_llm(
    user_text: str, conversation_history: List[Dict[str, str]] = None, timeout: float = 15.0
) -> Tuple[Dict[str, Any], List[str]]:
    """
    使用大模型进行实体抽取，从用户输入和对话历史中提取参数

    返回：(提取的参数字典, 缺失的字段列表)
    """
    # 【新增】优先检测班级场景，如果检测到班级，直接返回预填充的参数
    # 注意：这里假设是lesson_plan意图，因为只有课课练才支持班级检测
    is_class, class_params = detect_class_and_fill_params(user_text, intent="lesson_plan")
    if is_class:
        # 检测到班级，直接返回预填充的参数，missing=[]
        return class_params, []

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
    
    # 加载班级配置并生成班级配置文本
    class_profiles = load_class_profiles()
    class_profiles_text = ""
    if class_profiles:
        class_profiles_text = "如果用户提到以下班级，结合班级体测数据提取trained_weaknesses：\n"
        for class_name, profile in class_profiles.items():
            weaknesses = profile.get("trained_weaknesses", "")
            description = profile.get("description", "")
            class_profiles_text += f"## {class_name}核心薄弱维度：{weaknesses}\n"
            if "weakness_details" in profile:
                for weakness, detail in profile["weakness_details"].items():
                    class_profiles_text += f"- {weakness}：{detail}\n"
            class_profiles_text += "\n"
    
    # 加载参数提取用户提示词模板
    user_template = load_prompt_template("param_extraction_user")
    user = user_template.format(
        history_text=history_text if history_text else "（无历史记录）",
        user_text=user_text,
        class_profiles_text=class_profiles_text if class_profiles_text else "（无班级配置信息）"
    )

    resp = model.client.chat.completions.create(
        model=model.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=400,
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    # JSON截取
    start = content.find("{")
    end = content.rfind("}")
    parsed: Dict[str, Any] = {}
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
        except Exception:
            parsed = {}
    out = {
        "semantic_query": parsed.get("semantic_query") or None,
        "count_query": str(parsed.get("count_query")) if parsed.get("count_query") not in (None, "") else None,
        "grades_query": str(parsed.get("grades_query")) if parsed.get("grades_query") not in (None, "") else None,
        "trained_weaknesses": parsed.get("trained_weaknesses") or None,
        "top_k": int(parsed.get("top_k") or 5),
    }
    missing: List[str] = []
    for key in ["semantic_query", "count_query", "grades_query", "trained_weaknesses"]:
        if not out.get(key):
            missing.append(key)
    return out, missing


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 8.0) -> List[Dict[str, Any]]:
    # 记录请求信息
    if os.getenv('DEBUG_AI','1')=='1':
        logger.info(f"[TEACHER] 🚀 检索接口请求")
        logger.info(f"   URL: {url}")
        logger.info(f"   Timeout: {timeout}秒")
        logger.info(f"   Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    resp = requests.post(url, json=payload, timeout=timeout)

    # 记录响应信息
    if os.getenv('DEBUG_AI','1')=='1':
        logger.info(f"[TEACHER] ✅ 检索接口响应: status_code={resp.status_code}")

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


def call_hybrid_search(base_url, payload, timeout=8.0):
    url = base_url + "/extended-search/hybrid"
    return _post_json(url, payload, timeout)


def call_sports_meeting_search(base_url, payload, timeout=8.0):
    url = base_url + "/search/hybrid"
    return _post_json(url, payload, timeout)


def build_plan_messages(
    results: List[Dict[str, Any]],
    params: Dict[str, Any],
    user_text: str,
    need_guidance: bool = False,
) -> List[Dict[str, str]]:
    """
    构造用于生成备课方案的messages，供流式与非流式复用。

    参数：
        results: 检索结果列表（如果need_guidance=True，可以为空列表）
        params: 实体抽取的参数
        user_text: 用户原始输入
        need_guidance: 如果为True，生成引导语而不是方案

    返回：messages列表，可直接用于chat.completions.create
    """
    model = OptimizedAIModel()

    # 如果需要引导，生成引导提示
    if need_guidance:
        collected_str = json.dumps({
            "semantic_query": params.get("semantic_query") or "",
            "count_query": params.get("count_query") or "",
            "grades_query": params.get("grades_query") or "",
            "trained_weaknesses": params.get("trained_weaknesses") or "",
            "plan_type": params.get("plan_type") or "",
        }, ensure_ascii=False, indent=2)

        # 根据意图类型和已收集的实体判断缺失字段
        plan_type = params.get("plan_type")
        is_sports_meeting = plan_type == "sports_meeting"
        is_lesson_plan = plan_type == "lesson_plan"

        # 判断缺失字段
        missing_info = []
        if is_sports_meeting:
            # 全员运动会：需要操场条件、年级、人数等信息
            if not params.get("semantic_query"):
                missing_info.append("操场条件、跑道数量、场地规模")
            if not params.get("grades_query"):
                missing_info.append("参与年级")
            if not params.get("count_query"):
                missing_info.append("参与学生人数")
        elif is_lesson_plan:
            # 课课练：需要班级（grades_query）或弱项（trained_weaknesses），满足任一即可
            has_grades = bool(params.get("grades_query"))
            has_weaknesses = bool(params.get("trained_weaknesses"))
            if not has_grades and not has_weaknesses:
                missing_info.append("班级或薄弱项（如：速度、力量、柔韧等）")
            elif not has_grades:
                missing_info.append("班级信息")
            elif not has_weaknesses:
                missing_info.append("薄弱项（如：速度、力量、柔韧等）")

        missing_str = "、".join(missing_info) if missing_info else "无"

        # 加载引导语模板
        guidance_template = load_prompt_template("guidance_prompt")
        user_prompt = guidance_template.format(
            user_text=user_text,
            collected_info=collected_str,
            plan_type=plan_type or "未确定",
            missing_info=missing_str
        )
        return [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    # 正常生成方案
    # 汇总检索结果，控制上下文长度
    top_k = int(params.get("top_k") or 5)
    # 优先只取 text 字段；若缺失再回退到其他字段
    texts: List[str] = []
    for r in results[: top_k]:
        t = r.get("text")
        if t:
            texts.append(str(t).strip())
            continue
        # 回退：拼一个简要描述，尽量不丢关键信息
        title = r.get("title") or r.get("name") or ""
        desc = r.get("description") or r.get("desc") or ""
        media = r.get("image") or r.get("cover") or r.get("thumbnail") or r.get("media_url") or r.get("img") or ""
        fallback = "\n".join(x for x in [title, desc, media] if x) or ""
        if fallback:
            texts.append(fallback)
    results_text = "\n\n".join(texts) if texts else "无检索结果text，需由你结合参数生成通用方案。"

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
    is_chat = plan_type == "chat" or plan_type == ""

    if is_sports_meeting:
        # 全员运动会方案生成提示词
        template = load_prompt_template("plan_generation_sports_meeting")
        user_prompt = template.format(
            user_text=user_text,
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
        if not class_analysis_text and grades_query:
            class_analysis_text = f"   - 由于配置中没有该班级的详细信息，本方案将基于{grades_query}年级的一般特点提供通用的练习推荐。\n   - **重要**：请在方案开头展示这个提示信息！\n"

        template = load_prompt_template("plan_generation_lesson_plan")
        user_prompt = template.format(
            user_text=user_text,
            meta=json.dumps(meta, ensure_ascii=False, indent=2),
            results_text=results_text,
            class_analysis_text=class_analysis_text
        )
    elif plan_type == "chat" or plan_type == "":
        # 闲聊或未识别意图：仅返回系统提示与原始输入
        messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]
        if conversation_history := params.get("conversation_history"):
            for msg in conversation_history[-6:]:
                messages.append(msg)
        messages.append({"role": "user", "content": user_text})
        return messages
    else:
        # 未知意图，返回默认消息
        return [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    # 返回消息列表
    messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}]
    if conversation_history := params.get("conversation_history"):
        for msg in conversation_history[-6:]:
            messages.append(msg)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_plan_stream(
    results: List[Dict[str, Any]],
    params: Dict[str, Any],
    user_text: str,
    need_guidance: bool = False,
):
    """
    流式生成备课方案

    参数：
        results: 检索结果列表
        params: 实体抽取的参数
        user_text: 用户原始输入
        need_guidance: 如果为True，生成引导语而不是方案

    返回：生成器，逐块返回生成的文本
    """
    model = OptimizedAIModel()
    messages = build_plan_messages(results, params, user_text, need_guidance)

    # 调试信息已移除（前端可见模型回复内容）

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
        for event in stream:
            try:
                delta = event.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    chunk_count += 1
                    yield content
            except Exception:
                chunk = None
                try:
                    chunk = event["choices"][0]["delta"].get("content")
                except Exception:
                    pass
                if chunk:
                    chunk_count += 1
                    yield chunk
    except Exception as e:
        if os.getenv('DEBUG_AI','1')=='1':
            logger.error(f"[TEACHER] 流式生成失败: {e}")
        yield f"生成失败: {str(e)}"


def generate_plan(
    results: List[Dict[str, Any]],
    params: Dict[str, Any],
    user_text: str,
    need_guidance: bool = False,
) -> str:
    """
    非流式生成备课方案

    参数：
        results: 检索结果列表
        params: 实体抽取的参数
        user_text: 用户原始输入
        need_guidance: 如果为True，生成引导语而不是方案

    返回：生成的文本
    """
    model = OptimizedAIModel()
    messages = build_plan_messages(results, params, user_text, need_guidance)

    try:
        resp = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=3000,
            temperature=0.7,
            top_p=0.9,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"生成失败: {e}")
        return f"生成失败: {str(e)}"

