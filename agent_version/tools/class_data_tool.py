#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
班级数据工具
"""

import json
import logging
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from .base import load_class_profiles, ToolResult

logger = logging.getLogger("agent_tools")


class ClassProfileLookupInput(BaseModel):
    """班级配置查询输入"""
    class_name: Optional[str] = Field(
        default=None,
        description="班级名称，如 '一年级1班'。如果不提供则返回所有班级列表"
    )


class ClassProfileLookupTool(BaseTool):
    """
    班级配置查询工具
    
    查询班级的体测数据分析结果和薄弱项配置
    """
    name: str = "class_profile_lookup"
    description: str = """查询班级配置信息。
用于获取班级的体测数据分析结果，包括：
- 年级信息
- 薄弱项（速度、力量、柔韧等）
- 学生分组情况
- 薄弱项详细描述

如果不提供班级名称，返回所有已配置的班级列表。"""
    args_schema: Type[BaseModel] = ClassProfileLookupInput
    
    def _run(self, class_name: Optional[str] = None) -> str:
        """执行班级配置查询"""
        try:
            profiles = load_class_profiles()
            
            if not class_name:
                # 返回所有班级列表
                class_list = list(profiles.keys())
                return json.dumps({
                    "success": True,
                    "data": {
                        "class_list": class_list,
                        "count": len(class_list)
                    }
                }, ensure_ascii=False)
            
            # 查询特定班级
            if class_name in profiles:
                profile = profiles[class_name]
                return json.dumps({
                    "success": True,
                    "data": {
                        "class_name": class_name,
                        "profile": profile
                    }
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "success": False,
                    "data": None,
                    "error": f"未找到班级配置: {class_name}"
                }, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"班级配置查询失败: {e}")
            return json.dumps({"success": False, "data": None, "error": str(e)})


class ClassDataAnalysisInput(BaseModel):
    """班级数据分析输入"""
    class_name: str = Field(description="班级名称")
    analysis_type: Optional[str] = Field(
        default="weakness",
        description="分析类型: weakness(薄弱项分析) | grouping(学生分组)"
    )


class ClassDataAnalysisTool(BaseTool):
    """
    班级数据分析工具
    
    分析班级的体测数据，生成薄弱项分析和学生分组
    """
    name: str = "class_data_analysis"
    description: str = """分析班级体测数据。
用于获取班级的详细分析信息，包括：
- 薄弱项分析：识别班级整体的薄弱维度
- 学生分组：按薄弱项对学生进行分组

需要提供班级名称，返回分析结果。"""
    args_schema: Type[BaseModel] = ClassDataAnalysisInput
    
    def _run(
        self,
        class_name: str,
        analysis_type: str = "weakness"
    ) -> str:
        """执行班级数据分析"""
        try:
            profiles = load_class_profiles()
            
            if class_name not in profiles:
                return json.dumps({
                    "success": False,
                    "data": None,
                    "error": f"未找到班级配置: {class_name}"
                }, ensure_ascii=False)
            
            profile = profiles[class_name]
            
            if analysis_type == "weakness":
                # 返回薄弱项分析
                result = {
                    "class_name": class_name,
                    "grades_query": profile.get("grades_query", ""),
                    "trained_weaknesses": profile.get("trained_weaknesses", ""),
                    "weakness_details": profile.get("weakness_details", {}),
                    "description": profile.get("description", "")
                }
            elif analysis_type == "grouping":
                # 返回学生分组
                result = {
                    "class_name": class_name,
                    "student_groups": profile.get("student_groups", {})
                }
            else:
                # 返回完整配置
                result = {
                    "class_name": class_name,
                    "profile": profile
                }
            
            return json.dumps({
                "success": True,
                "data": result
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"班级数据分析失败: {e}")
            return json.dumps({"success": False, "data": None, "error": str(e)})

