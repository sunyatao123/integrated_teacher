#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模型调用模块 - 优化版本
"""

import os
import json
import time
from typing import Dict, List, Optional
from openai import OpenAI

class OptimizedAIModel:
    def __init__(self):
        """初始化AI模型"""
        self.api_key = os.getenv(
            "SILICONFLOW_API_KEY",
            "sk-iwcqksidcwhiasawyqkctbeydcqkylwynkdypvbuzmhtvies"
        )
        self.base_url = os.getenv(
            "SILICONFLOW_BASE_URL", 
            "https://api.siliconflow.cn/v1"
        )
        self.model = os.getenv(
            "SILICONFLOW_MODEL",
            "deepseek-ai/DeepSeek-V3"
        )
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 系统提示词
        self.system_prompt = """你是一个专业的AI健康助教，名字叫"小乐"。你的职责是：

1. 根据学生的体测数据分析其健康状况。你的用户是小孩子，你语气好亲切友好，不需要用您，
2. 提供个性化的运动建议和训练指导，当用户询问，你是否能生成视频等，你需要回答，我不能直接生成视频，但我可以为您查找，如果您需要，我可以为您推荐视频，并基于该学生的体测数据的弱项，推荐视频，最后说请稍等，正在为你推荐。
3. 识别用户的意图，判断是否需要推荐视频，严格按照：推荐视频，在最后加上句“请稍等，正在为您推荐视频”，不推荐视频，不要加上这句话。注意：当用户输入“查看我的整体情况，判断为不推荐视频”
4. 根据推荐的动作列表给出训练计划，并给出训练时间、训练强度、训练频率等建议。
5. 对于一些用户情绪沮丧的话，如：我跑不过别人怎么办等，试着鼓励。无需询问是否需要推荐视频。
6. 用温暖、鼓励的语气与学生交流。你的口吻要像一个健康助教，而不是一个机器人。在必要时给予孩子鼓励，让孩子感到被关心和被支持。
7. 基于科学依据给出专业建议，对于非体育学科的问题，如用户问你某道数学题怎么解，你要学会巧妙的回避。

请始终保持积极正面的态度，用简洁易懂的语言解释专业概念，返回markdown格式。"""

        # 缓存机制
        self._intent_cache = {}
        self._response_cache = {}
        self._cache_ttl = 300  # 5分钟缓存

    def summarize_with_citations(self, query: str, sources: List[Dict]) -> Dict:
        """
        基于联网搜索结果生成带来源标注的总结。
        返回：{ summary: str, sentences: [{text, source_id}], sources: [...] }
        """
        try:
            # 构造来源摘要文本，供LLM引用
            lines = []
            for s in sources[:8]:
                lines.append(f"[{s['id']}] 标题: {s['title']}\nURL: {s['url']}\n摘录: {s.get('content','')[:800]}")
            source_text = "\n\n".join(lines)

            prompt = f"""
你是一个专业的AI健康助教，名字叫"小乐"，用温暖、鼓励的语气与学生交流
基于科学依据给出专业建议，对于非体育学科的问题，如用户问你某道数学题怎么解，你要学会巧妙的回避。
请基于以下来源为用户的查询生成结构化总结，并在每一句话末尾标注来源编号，如[1]、[2]。若一句话综合多个来源，可使用[1,3]。

用户查询：{query}

可用来源（编号对应来源）：
{source_text}

要求：
1. 先给出要点式总结，3-6条，每条一句话，以"- "开头，句末标注来源编号。
2. 若存在不一致或争议，明确指出。
3. 在最后给出行动建议，1-3条，句末也要标注来源编号。
4. 只使用提供的来源信息，不要编造。
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是严谨的研究助理，回答必须可溯源。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=900,
                temperature=0.2
            )

            text = response.choices[0].message.content.strip()

            # 粗粒度拆句并提取来源编号
            sentences = []
            for line in text.split("\n"):
                t = line.strip()
                if not t:
                    continue
                # 寻找类似 [1] 或 [1,2] 的尾注
                src_ids = []
                import re
                m = re.search(r"\[(\d+(?:\s*,\s*\d+)*)\]\s*$", t)
                if m:
                    src_ids = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()]
                sentences.append({"text": t, "source_id": src_ids})

            return {"summary": text, "sentences": sentences, "sources": sources}
        except Exception as e:
            return {"summary": f"生成总结失败：{e}", "sentences": [], "sources": sources}

    def generate_response_with_recommendations(self, 
                                             user_message: str, 
                                             student_analysis: Dict = None,
                                             recommended_actions: List[Dict] = None,
                                             conversation_history: List[Dict] = None) -> Dict:
        """
        一次性生成回复和推荐，减少API调用次数
        
        Returns:
            {
                'message': str,
                'need_recommendations': bool,
                'question_type': str,
                'ai_suggestions': List[str]
            }
        """
        try:
            # 构建上下文信息
            context = self._build_context(student_analysis, recommended_actions)
            
            # 构建消息列表
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 添加上下文信息
            if context:
                messages.append({"role": "system", "content": context})
            
            # 添加对话历史（扩大到最近8条，提供更完整语境）
            if conversation_history:
                for msg in conversation_history[-8:]:
                    messages.append(msg)
            
            # 增强的提示词，让AI一次性完成所有任务
            enhanced_prompt = f"""请回答用户问题，并在最后以JSON格式返回以下信息：
1. need_recommendations: 是否需要推荐训练动作 (true/false)，当用户输入“查看我的整体情况”，判断为不推荐视频，返回false。对于一些用户情绪比较低落的话，如：我跑不过别人怎么办等，试着鼓励，不推荐视频，返回false。对于用户询问，能否生成视频等，请仔细鉴别。
2. question_type: 问题类型 (speed/endurance/strength/flexibility/jumping/coordination/overall/general)
3. ai_suggestions: 3个相关的联想问题列表
4. recommended_actions: 若 need_recommendations=true，请给出不超过6个“动作名称”的列表（仅名称，按优先顺序排列）。

用户问题: {user_message}

请先给出你的回答，然后在最后添加：
```json
{{
    "need_recommendations": true/false,
    "question_type": "类型",
    "ai_suggestions": ["问题1", "问题2", "问题3"],
    "recommended_actions": ["动作A", "动作B"]
}}
```"""
            
            messages.append({"role": "user", "content": enhanced_prompt})
            
            # 调用AI模型
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1200,  # 增加token限制
                temperature=0.7,
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 解析回复和JSON信息
            message, metadata = self._parse_response_with_metadata(response_text)
            
            return {
                'message': message,
                'need_recommendations': metadata.get('need_recommendations', False),
                'question_type': metadata.get('question_type', 'general'),
                'ai_suggestions': metadata.get('ai_suggestions', []),
                'recommended_actions': metadata.get('recommended_actions', [])
            }
            
        except Exception as e:
            print(f"AI模型调用失败: {e}")
            return {
                'message': self._get_fallback_response(user_message, student_analysis),
                'need_recommendations': False,
                'question_type': 'general',
                'ai_suggestions': []
            }

    def generate_response_stream_optimized(self,
                                         user_message: str,
                                         student_analysis: Dict = None,
                                         recommended_actions: List[Dict] = None,
                                         conversation_history: List[Dict] = None):
        """
        优化的流式回复生成
        """
        try:
            context = self._build_context(student_analysis, recommended_actions)

            messages = [{"role": "system", "content": self.system_prompt}]
            if context:
                messages.append({"role": "system", "content": context})
            if conversation_history:
                for msg in conversation_history[-8:]:
                    messages.append(msg)
            messages.append({"role": "user", "content": user_message})

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=800,  # 减少token限制，提高速度
                temperature=0.7,
                top_p=0.9,
                stream=True
            )

            for event in stream:
                try:
                    delta = event.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield content
                except Exception:
                    chunk = None
                    try:
                        chunk = event["choices"][0]["delta"].get("content")
                    except Exception:
                        pass
                    if chunk:
                        yield chunk
        except Exception as e:
            print(f"AI模型流式调用失败: {e}")
            fallback = self._get_fallback_response(user_message, student_analysis)
            if fallback:
                yield fallback

    def _parse_response_with_metadata(self, response_text: str) -> tuple:
        """解析包含元数据的回复"""
        try:
            # 查找JSON部分
            json_start = response_text.find('```json')
            if json_start != -1:
                json_start += 7  # 跳过 ```json
                json_end = response_text.find('```', json_start)
                if json_end != -1:
                    json_text = response_text[:json_start-7].strip()  # 回复部分
                    metadata_text = response_text[json_start:json_end].strip()  # JSON部分
                    
                    try:
                        metadata = json.loads(metadata_text)
                        return json_text, metadata
                    except json.JSONDecodeError:
                        pass
            
            # 如果没有找到JSON，尝试查找普通JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_text = response_text[:json_start].strip()
                metadata_text = response_text[json_start:json_end+1]
                
                try:
                    metadata = json.loads(metadata_text)
                    return json_text, metadata
                except json.JSONDecodeError:
                    pass
            
            # 如果都没有找到，返回原文本
            return response_text, {}
            
        except Exception as e:
            print(f"解析回复元数据失败: {e}")
            return response_text, {}

    def _build_context(self, student_analysis: Dict = None, recommended_actions: List[Dict] = None) -> str:
        """构建上下文信息"""
        context_parts = []
        
        if student_analysis:
            context_parts.append("学生体测数据分析结果：")
            context_parts.append(f"- 总分：{student_analysis.get('total_score', 0)}分")
            context_parts.append(f"- 整体水平：{student_analysis.get('overall_assessment', '未知')}")
            context_parts.append(f"- 性别：{student_analysis.get('gender', '未知')}")
            context_parts.append(f"- 年级：{student_analysis.get('grade', '未知')}")
            
            if student_analysis.get('weak_items'):
                context_parts.append("- 需要加强的项目：")
                for item in student_analysis['weak_items']:
                    context_parts.append(f"  * {item['item']}：{item['score']}分（{item['level']}）")
            
            if student_analysis.get('scores'):
                context_parts.append("- 各项体测成绩：")
                for item, score in student_analysis['scores'].items():
                    context_parts.append(f"  * {item}：{score}分")
        
        if recommended_actions:
            context_parts.append("\n推荐训练动作：")
            for i, action in enumerate(recommended_actions[:3], 1):  # 只显示前3个
                context_parts.append(f"{i}. {action['action_name']}")
                context_parts.append(f"   - 说明：{action['description']}")
                context_parts.append(f"   - 训练方案：{action['sets']}组，每组{action['duration']}，休息{action['rest_time']}秒")
                context_parts.append(f"   - 针对素质：{action['target_quality']}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _get_fallback_response(self, user_message: str, student_analysis: Dict = None) -> str:
        """获取备用回复"""
        message_lower = user_message.lower()
        
        if any(keyword in message_lower for keyword in ['速度', '跑', '快']):
            return "关于提高跑步速度，建议进行间歇跑训练，加强腿部力量，保持正确的跑步姿势。"
        elif any(keyword in message_lower for keyword in ['耐力', '持久', '长跑']):
            return "提高耐力需要循序渐进的有氧训练，建议进行长距离慢跑和变速跑练习。"
        elif any(keyword in message_lower for keyword in ['力量', '肌肉', '引体']):
            return "力量训练建议从自重训练开始，逐步增加强度，注意动作标准性。"
        elif any(keyword in message_lower for keyword in ['柔韧', '拉伸', '灵活']):
            return "柔韧性训练需要每天坚持拉伸练习，动作要缓慢到位，保持呼吸顺畅。"
        else:
            return "我理解你的问题，建议你告诉我具体想了解哪个方面的训练，我会为你提供更详细的指导。"

    def generate_training_plan(self, student_analysis: Dict, recommended_actions: List[Dict]) -> str:
        """生成个性化训练计划"""
        try:
            prompt = f"""
基于以下学生体测分析结果，生成一个详细的个性化训练计划：

学生信息：
- 总分：{student_analysis.get('total_score', 0)}分
- 整体水平：{student_analysis.get('overall_assessment', '未知')}
- 需要加强的项目：{[item['item'] for item in student_analysis.get('weak_items', [])]}

推荐动作：{[action['action_name'] for action in recommended_actions[:5]]}

请生成一个包含以下内容的训练计划：
1. 训练目标
2. 每周训练安排
3. 具体训练内容
4. 注意事项
5. 预期效果

请用温暖鼓励的语气，提供实用的建议。
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"生成训练计划失败: {e}")
            return self._get_default_training_plan(student_analysis, recommended_actions)
    
    def _get_default_training_plan(self, student_analysis: Dict, recommended_actions: List[Dict]) -> str:
        """获取默认训练计划"""
        plan = f"基于你的体测分析（总分{student_analysis.get('total_score', 0)}分），我为你制定以下训练计划：\n\n"
        
        plan += "🎯 训练目标：\n"
        if student_analysis.get('weak_items'):
            for item in student_analysis['weak_items'][:3]:
                plan += f"- 提升{item['item']}成绩\n"
        else:
            plan += "- 保持现有水平，全面发展\n"
        
        plan += "\n📅 每周训练安排：\n"
        plan += "- 周一、三、五：力量训练\n"
        plan += "- 周二、四：有氧训练\n"
        plan += "- 周六：柔韧性训练\n"
        plan += "- 周日：休息\n"
        
        plan += "\n💪 推荐动作：\n"
        for i, action in enumerate(recommended_actions[:3], 1):
            plan += f"{i}. {action['action_name']} - {action['sets']}组×{action['duration']}\n"
        
        plan += "\n⚠️ 注意事项：\n"
        plan += "- 循序渐进，不要急于求成\n"
        plan += "- 注意动作标准性\n"
        plan += "- 保证充足休息和营养\n"
        plan += "- 如有不适立即停止\n"
        
        return plan
