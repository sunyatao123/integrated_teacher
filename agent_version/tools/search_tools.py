"""
检索工具模块 - 使用 @tool 装饰器定义
"""
import json
import logging
import os
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool

# 获取日志器（与agent_runner共用）
logger = logging.getLogger("Agent")

SEARCH_BASE_URL = os.getenv("SEARCH_BASE_URL", "http://127.0.0.1:8001")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_URL = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")


@tool
def lesson_plan_search(
    grades_query: str,
    trained_weaknesses: str = "",
    semantic_query: str = "",
    top_k: int = 5
) -> str:
    """
    课课练内部检索工具。搜索适合特定年级和薄弱项的训练动作和练习方法。
    
    使用场景：当用户需要日常体育课训练内容、体能训练方案、改善某项体能薄弱项时使用。
    
    Args:
        grades_query: 年级，如 "1", "3", "6"
        trained_weaknesses: 薄弱项，如 "速度", "力量", "柔韧", "耐力"
        semantic_query: 语义查询（可选），如"跳绳训练"、"球类运动"
        top_k: 返回结果数量
    
    Returns:
        检索结果的JSON字符串
    """
    url = SEARCH_BASE_URL + "/extended-search/hybrid"
    payload = {
        "semantic_query": semantic_query or trained_weaknesses or "体能训练",
        "count_query": "",
        "grades_query": str(grades_query),
        "trained_weaknesses": trained_weaknesses or "",
        "top_k": top_k,
    }
    
    logger.debug(f"[lesson_plan_search] URL: {url}")
    logger.debug(f"[lesson_plan_search] Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        resp = requests.post(url, json=payload, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []
        
        results = results[:top_k] if results else []
        
        logger.debug(f"[lesson_plan_search] 返回 {len(results)} 条结果")
        
        return json.dumps({
            "success": True,
            "source": "internal_lesson_plan",
            "count": len(results),
            "results": results
        }, ensure_ascii=False)
        
    except requests.exceptions.Timeout:
        logger.warning("[lesson_plan_search] 请求超时")
        return json.dumps({"success": False, "source": "internal_lesson_plan", "error": "检索超时", "results": []})
    except requests.exceptions.ConnectionError:
        logger.warning("[lesson_plan_search] 连接失败")
        return json.dumps({"success": False, "source": "internal_lesson_plan", "error": "检索服务不可用", "results": []})
    except Exception as e:
        logger.error(f"[lesson_plan_search] 异常: {str(e)}")
        return json.dumps({"success": False, "source": "internal_lesson_plan", "error": str(e), "results": []})


@tool
def sports_meeting_search(
    semantic_query: str,
    grades_query: str = "",
    count_query: str = "",
    top_k: int = 5
) -> str:
    """
    全员运动会内部检索工具。搜索运动会项目、趣味比赛、班级对抗等内容。
    
    使用场景：当用户需要运动会方案、比赛项目、趣味运动、班级对抗赛等内容时使用。
    
    Args:
        semantic_query: 项目描述，如 "球类项目"、"接力赛"、"趣味比赛"、"团队协作"
        grades_query: 年级（可选）
        count_query: 参与人数（可选）
        top_k: 返回结果数量
    
    Returns:
        检索结果的JSON字符串
    """
    url = SEARCH_BASE_URL + "/search/hybrid"
    payload = {
        "semantic_query": semantic_query,
        "count_query": str(count_query) if count_query else "",
        "grades_query": str(grades_query) if grades_query else "",
        "top_k": top_k,
    }
    
    logger.debug(f"[sports_meeting_search] URL: {url}")
    logger.debug(f"[sports_meeting_search] Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        resp = requests.post(url, json=payload, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []
        
        results = results[:top_k] if results else []
        
        logger.debug(f"[sports_meeting_search] 返回 {len(results)} 条结果")
        
        return json.dumps({
            "success": True,
            "source": "internal_sports_meeting",
            "count": len(results),
            "results": results
        }, ensure_ascii=False)
        
    except requests.exceptions.Timeout:
        logger.warning("[sports_meeting_search] 请求超时")
        return json.dumps({"success": False, "source": "internal_sports_meeting", "error": "检索超时", "results": []})
    except requests.exceptions.ConnectionError:
        logger.warning("[sports_meeting_search] 连接失败")
        return json.dumps({"success": False, "source": "internal_sports_meeting", "error": "检索服务不可用", "results": []})
    except Exception as e:
        logger.error(f"[sports_meeting_search] 异常: {str(e)}")
        return json.dumps({"success": False, "source": "internal_sports_meeting", "error": str(e), "results": []})


@tool
def web_search(query: str, top_k: int = 5) -> str:
    """
    互联网搜索工具（Tavily）。当内部检索无结果或结果不足时使用。
    
    使用场景：内部检索没有找到相关内容时，使用此工具从互联网获取补充信息。
    
    Args:
        query: 搜索查询词，如 "小学六年级 体育课课练 耐力训练方法"
        top_k: 返回结果数量
    
    Returns:
        搜索结果的JSON字符串
    """
    logger.debug(f"[web_search] Query: {query}")
    
    if not TAVILY_API_KEY:
        logger.warning("[web_search] Tavily API Key 未配置")
        return json.dumps({"success": False, "source": "web", "error": "Tavily API Key未配置", "results": []})
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": top_k,
    }
    
    try:
        resp = requests.post(
            TAVILY_API_URL,
            json=payload,
            timeout=15.0,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        
        results: List[Dict[str, Any]] = []
        
        answer = data.get("answer", "")
        if answer:
            results.append({
                "text": answer,
                "title": "搜索摘要",
                "source": "tavily_answer",
                "url": "",
            })
        
        for item in data.get("results", [])[:top_k]:
            content = item.get("content", "")
            if content:
                results.append({
                    "text": content,
                    "title": item.get("title", ""),
                    "source": "tavily",
                    "url": item.get("url", ""),
                })
        
        logger.debug(f"[web_search] 返回 {len(results)} 条结果")
        
        return json.dumps({
            "success": True,
            "source": "web",
            "count": len(results),
            "results": results
        }, ensure_ascii=False)
        
    except requests.exceptions.Timeout:
        logger.warning("[web_search] 请求超时")
        return json.dumps({"success": False, "source": "web", "error": "搜索超时", "results": []})
    except Exception as e:
        logger.error(f"[web_search] 异常: {str(e)}")
        return json.dumps({"success": False, "source": "web", "error": str(e), "results": []})


@tool
def load_class_profile(class_name: str) -> str:
    """
    加载班级画像工具。获取指定班级的体测分析和学生分组信息。
    
    使用场景：当用户提到具体班级名称，需要根据班级实际情况定制方案时使用。
    
    Args:
        class_name: 班级名称，如 "六年级5班"
    
    Returns:
        班级画像信息的JSON字符串
    """
    logger.debug(f"[load_class_profile] 班级: {class_name}")
    
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "class_profiles.json"),
        os.path.join(os.path.dirname(__file__), "..", "prompts", "class_profiles.json"),
        os.path.join(os.getcwd(), "prompts", "class_profiles.json"),
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
    
    if not path:
        logger.warning("[load_class_profile] 班级配置文件不存在")
        return json.dumps({"success": False, "error": "班级配置文件不存在", "profile": {}})
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        
        profile = profiles.get(class_name, {})
        if profile:
            logger.debug(f"[load_class_profile] 找到班级配置: {class_name}")
            return json.dumps({"success": True, "class_name": class_name, "profile": profile}, ensure_ascii=False)
        else:
            logger.debug(f"[load_class_profile] 未找到班级: {class_name}")
            return json.dumps({"success": False, "class_name": class_name, "error": "未找到该班级配置", "profile": {}})
    except Exception as e:
        logger.error(f"[load_class_profile] 异常: {str(e)}")
        return json.dumps({"success": False, "error": str(e), "profile": {}})


# 导出所有工具
ALL_TOOLS = [
    lesson_plan_search,
    sports_meeting_search,
    web_search,
    load_class_profile,
]
