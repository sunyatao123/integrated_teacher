#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索工具
"""

import json
import logging
import requests
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from .base import ToolResult
from agent_version.config import SEARCH_BASE_URL, SEARCH_TIMEOUT

logger = logging.getLogger("agent")


class LessonPlanSearchInput(BaseModel):
    """课课练检索工具输入"""
    grades_query: str = Field(description="年级，如 '1', '3', '5'")
    trained_weaknesses: str = Field(description="薄弱项，如 '速度', '力量', '柔韧'")
    semantic_query: Optional[str] = Field(default="", description="语义查询关键词")
    top_k: Optional[int] = Field(default=10, description="返回结果数量")


class LessonPlanSearchTool(BaseTool):
    """
    课课练内容检索工具
    
    搜索适合特定年级和薄弱项的训练动作和练习方法
    """
    name: str = "lesson_plan_search"
    description: str = """搜索课课练训练内容。
用于查找适合特定年级和薄弱项的训练动作和练习方法。
需要提供：
- grades_query: 年级（必填）
- trained_weaknesses: 薄弱项（必填）
- semantic_query: 可选的语义查询
- top_k: 返回结果数量（默认10）"""
    args_schema: Type[BaseModel] = LessonPlanSearchInput
    
    def _run(
        self,
        grades_query: str,
        trained_weaknesses: str,
        semantic_query: str = "",
        top_k: int = 10
    ) -> str:
        """执行课课练检索"""
        try:
            url = f"{SEARCH_BASE_URL}/extended-search/hybrid"
            payload = {
                "semantic_query": semantic_query or "",
                "count_query": "",
                "grades_query": grades_query,
                "trained_weaknesses": trained_weaknesses,
                "top_k": top_k
            }
            
            resp = requests.post(url, json=payload, timeout=SEARCH_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", data) if isinstance(data, dict) else data
            
            return json.dumps({
                "success": True,
                "data": results,
                "count": len(results)
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"课课练检索失败: {e}")
            return json.dumps({"success": False, "data": [], "error": str(e)})


class SportsMeetingSearchInput(BaseModel):
    """运动会检索工具输入"""
    semantic_query: str = Field(description="操场条件描述，如 '标准操场 8条跑道'")
    grades_query: Optional[str] = Field(default="", description="年级")
    count_query: Optional[str] = Field(default="", description="学生人数")
    top_k: Optional[int] = Field(default=5, description="返回结果数量")


class SportsMeetingSearchTool(BaseTool):
    """
    全员运动会内容检索工具
    
    搜索适合特定操场条件和人数的运动会项目
    """
    name: str = "sports_meeting_search"
    description: str = """搜索全员运动会项目内容。
用于查找适合特定操场条件、年级和人数的运动会项目。
需要提供：
- semantic_query: 操场条件描述（必填）
- grades_query: 年级（可选）
- count_query: 学生人数（可选）
- top_k: 返回结果数量（默认5）"""
    args_schema: Type[BaseModel] = SportsMeetingSearchInput
    
    def _run(
        self,
        semantic_query: str,
        grades_query: str = "",
        count_query: str = "",
        top_k: int = 5
    ) -> str:
        """执行运动会检索"""
        try:
            url = f"{SEARCH_BASE_URL}/search/hybrid"
            payload = {
                "semantic_query": semantic_query,
                "count_query": count_query,
                "grades_query": grades_query,
                "top_k": top_k
            }
            
            resp = requests.post(url, json=payload, timeout=SEARCH_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", data) if isinstance(data, dict) else data
            
            return json.dumps({
                "success": True,
                "data": results,
                "count": len(results)
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"运动会检索失败: {e}")
            return json.dumps({"success": False, "data": [], "error": str(e)})

