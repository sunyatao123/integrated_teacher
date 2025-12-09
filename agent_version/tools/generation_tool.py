#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案生成工具 - 支持流式输出
"""

import json
import logging
from typing import Optional, Type, Dict, Any, List, Generator
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from .base import AIModelClient, load_prompt_template, load_class_profiles, ToolResult

logger = logging.getLogger("agent")


class PlanGenerationInput(BaseModel):
    """方案生成工具输入"""
    plan_type: str = Field(description="方案类型: lesson_plan/sports_meeting/chat")
    user_text: str = Field(description="用户原始输入")
    params: Dict[str, Any] = Field(description="提取的参数")
    search_results: Optional[List[Dict]] = Field(default=None, description="检索结果")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="对话历史"
    )
    need_guidance: Optional[bool] = Field(
        default=False, description="是否需要生成引导语（参数不全时）"
    )
    missing_fields: Optional[List[str]] = Field(
        default=None, description="缺失的字段列表"
    )


class PlanGenerationTool(BaseTool):
    """
    方案生成工具
    
    根据用户需求和检索结果生成备课方案
    """
    name: str = "plan_generation"
    description: str = """生成备课方案或引导语。

根据用户需求、提取的参数和检索结果，生成：
- 课课练方案：包含训练动作、分组练习等
- 运动会方案：包含项目安排、场地布置等
- 引导语：当参数不全时，引导用户补充信息

需要提供：
- plan_type: 方案类型
- user_text: 用户原始输入
- params: 提取的参数
- search_results: 检索结果（可选）
- need_guidance: 是否生成引导语"""
    args_schema: Type[BaseModel] = PlanGenerationInput
    
    def _run(
        self,
        plan_type: str,
        user_text: str,
        params: Dict[str, Any],
        search_results: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        need_guidance: bool = False,
        missing_fields: Optional[List[str]] = None
    ) -> str:
        """执行方案生成（非流式）"""
        try:
            result = self._generate_plan(
                plan_type, user_text, params,
                search_results or [],
                conversation_history or [],
                need_guidance,
                missing_fields or []
            )
            return result
        except Exception as e:
            logger.error(f"方案生成失败: {e}")
            return f"生成失败: {str(e)}"

    def run_stream(
        self,
        plan_type: str,
        user_text: str,
        params: Dict[str, Any],
        search_results: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        need_guidance: bool = False,
        missing_fields: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """流式方案生成"""
        try:
            for chunk in self._generate_plan_stream(
                plan_type, user_text, params,
                search_results or [],
                conversation_history or [],
                need_guidance,
                missing_fields or []
            ):
                yield chunk
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"生成失败: {str(e)}"
    
    def _build_results_text(self, results: List[Dict], top_k: int = 5) -> str:
        """构建检索结果文本"""
        texts = []
        for r in results[:top_k]:
            t = r.get("text")
            if t:
                texts.append(str(t).strip())
                continue
            title = r.get("title") or r.get("name") or ""
            desc = r.get("description") or r.get("desc") or ""
            media = r.get("image") or r.get("media_url") or ""
            fallback = "\n".join(x for x in [title, desc, media] if x)
            if fallback:
                texts.append(fallback)
        return "\n\n".join(texts) if texts else "无检索结果，需由你结合参数生成通用方案。"
    
    def _build_class_analysis_text(self, params: Dict) -> str:
        """构建班级分析文本"""
        detected_class = params.get("detected_class_name")
        if not detected_class:
            return ""
        
        profiles = load_class_profiles()
        if detected_class not in profiles:
            return ""
        
        profile = profiles[detected_class]
        text = ""
        
        # 薄弱项详情
        weakness_details = profile.get("weakness_details", {})
        if weakness_details:
            text = f"   - 班级：{detected_class}\n   - 班级薄弱项描述：\n"
            for weakness, detail in weakness_details.items():
                text += f"     * {weakness}：{detail}\n"
        
        # 学生分组
        student_groups = profile.get("student_groups", {})
        if student_groups:
            text += f"\n   - {detected_class}学生分组情况：\n"
            for group_key, group_info in student_groups.items():
                count = group_info.get("count", 0)
                weakness_items = group_info.get("weakness_items", [])
                student_details = group_info.get("student_details", [])
                
                text += f"     * {group_key}薄弱组：{count}人\n"
                if weakness_items:
                    text += f"       薄弱项目：{', '.join(weakness_items)}\n"
                if student_details:
                    text += f"       学生名单：\n"
                    for student in student_details:
                        name = student.get("姓名", f"学生{student.get('序号', '')}")
                        sid = student.get("学生编号", "")
                        text += f"         • {name} [{sid}]\n"
            
            text += "\n   - **重要**：请在方案开头展示上述学生分组情况！\n"
        
        return text
    
    def _build_messages(
        self,
        plan_type: str,
        user_text: str,
        params: Dict[str, Any],
        results: List[Dict],
        conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """构建LLM消息列表"""
        system_prompt = load_prompt_template("teacher_system_prompt")
        results_text = self._build_results_text(results, params.get("top_k", 10))
        meta = json.dumps(params, ensure_ascii=False, indent=2)

        if plan_type == "lesson_plan":
            template = load_prompt_template("plan_generation_lesson_plan")
            class_analysis = self._build_class_analysis_text(params)
            user_prompt = template.format(
                user_text=user_text,
                conversation_history=conversation_history,
                meta=meta,
                results_text=results_text,
                class_analysis_text=class_analysis
            )
        else:  # sports_meeting
            template = load_prompt_template("plan_generation_sports_meeting")
            user_prompt = template.format(
                user_text=user_text,
                conversation_history=conversation_history,
                meta=meta,
                results_text=results_text,
                grades_query=params.get("grades_query", ""),
                count_query=params.get("count_query", ""),
                semantic_query=params.get("semantic_query", "标准操场")
            )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _generate_plan(
        self,
        plan_type: str,
        user_text: str,
        params: Dict[str, Any],
        results: List[Dict],
        conversation_history: List[Dict[str, str]],
        need_guidance: bool,
        missing: List[str]
    ) -> str:
        """生成方案（非流式）"""
        client = AIModelClient()
        system_prompt = load_prompt_template("teacher_system_prompt")

        if need_guidance:
            return self._generate_guidance(client, params, user_text, missing, system_prompt)

        if plan_type == "chat":
            return self._generate_chat(client, user_text, conversation_history, system_prompt)

        messages = self._build_messages(plan_type, user_text, params, results, conversation_history)

        resp = client.client.chat.completions.create(
            model=client.model,
            messages=messages,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.9,
        )
        return resp.choices[0].message.content.strip()

    def _generate_plan_stream(
        self,
        plan_type: str,
        user_text: str,
        params: Dict[str, Any],
        results: List[Dict],
        conversation_history: List[Dict[str, str]],
        need_guidance: bool,
        missing: List[str]
    ) -> Generator[str, None, None]:
        """流式生成方案"""
        client = AIModelClient()
        system_prompt = load_prompt_template("teacher_system_prompt")

        # 引导语和闲聊使用流式生成
        if need_guidance:
            for chunk in self._generate_guidance_stream(client, params, user_text, missing, system_prompt):
                yield chunk
            return

        if plan_type == "chat":
            for chunk in self._generate_chat_stream(client, user_text, conversation_history, system_prompt):
                yield chunk
            return

        messages = self._build_messages(plan_type, user_text, params, results, conversation_history)

        # 流式调用LLM，并在服务器侧进行适度缓冲，避免前端收到过多极小 token
        stream = client.client.chat.completions.create(
            model=client.model,
            messages=messages,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )

        buffer = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                buffer += token
                # 累积到一定长度或出现换行符再向前端推送，减轻前端渲染压力
                if "\n" in buffer or len(buffer) >= 60:
                    yield buffer
                    buffer = ""

        if buffer:
            # 将剩余内容一次性推送
            yield buffer

    def _build_guidance_messages(self, params: Dict, user_text: str, missing: List[str], system_prompt: str) -> List[Dict]:
        """构建引导语消息"""
        missing_info = []
        if "semantic_query" in missing:
            missing_info.append("学生人数、年级、操场大小、跑道数量、操场条件")
        if "grades_query" in missing and "trained_weaknesses" in missing:
            missing_info.append("年级和薄弱项（如：形态、耐力、力量、柔韧、速度、机能等）")
        elif "grades_query" in missing:
            missing_info.append("年级信息")
        elif "trained_weaknesses" in missing:
            missing_info.append("薄弱项（如：形态、耐力、力量、柔韧、速度、机能等）")

        missing_str = "、".join(missing_info) if missing_info else "无"
        guidance_template = load_prompt_template("guidance_prompt")
        user_prompt = guidance_template.format(
            user_text=user_text,
            collected_info=params,
            plan_type=params.get("plan_type", "未确定"),
            missing_info=missing_str
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _generate_guidance(self, client: AIModelClient, params: Dict, user_text: str, missing: List[str], system_prompt: str) -> str:
        """生成引导语（非流式）"""
        messages = self._build_guidance_messages(params, user_text, missing, system_prompt)
        resp = client.client.chat.completions.create(
            model=client.model,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    def _generate_guidance_stream(self, client: AIModelClient, params: Dict, user_text: str, missing: List[str], system_prompt: str) -> Generator[str, None, None]:
        """流式生成引导语"""
        messages = self._build_guidance_messages(params, user_text, missing, system_prompt)
        stream = client.client.chat.completions.create(
            model=client.model,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_chat_messages(self, user_text: str, conversation_history: List[Dict[str, str]], system_prompt: str) -> List[Dict]:
        """构建闲聊消息"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _generate_chat(self, client: AIModelClient, user_text: str, conversation_history: List[Dict[str, str]], system_prompt: str) -> str:
        """生成闲聊回复（非流式）"""
        messages = self._build_chat_messages(user_text, conversation_history, system_prompt)
        resp = client.client.chat.completions.create(
            model=client.model,
            messages=messages,
            max_tokens=5000,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    def _generate_chat_stream(self, client: AIModelClient, user_text: str, conversation_history: List[Dict[str, str]], system_prompt: str) -> Generator[str, None, None]:
        """流式生成闲聊回复"""
        messages = self._build_chat_messages(user_text, conversation_history, system_prompt)
        stream = client.client.chat.completions.create(
            model=client.model,
            messages=messages,
            max_tokens=5000,
            temperature=0.7,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

