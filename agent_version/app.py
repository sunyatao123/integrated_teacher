#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能Agent版本 - Flask应用入口

核心特性：
1. 目标导向的智能决策
2. 动态策略生成与选择
3. 弹性执行与实时调整
4. 经验积累与持续学习
"""

import os
import sys
import json
import logging
from pathlib import Path

# 配置日志 - 必须在导入其他模块之前设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# 降低第三方库的日志级别
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("agent")

from flask import Flask, request, jsonify, Response, render_template, stream_with_context

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_version.config import (
    TEMPLATES_DIR,
    ORIGINAL_PROJECT_DIR,
)
from agent_version.agent import TeacherAgent

# 创建Flask应用
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(ORIGINAL_PROJECT_DIR / "static")
)

# 智能Agent实例缓存（按session_id）
_agents: dict = {}


def get_agent(session_id: str = "default") -> TeacherAgent:
    """获取或创建智能Agent实例"""
    if session_id not in _agents:
        _agents[session_id] = TeacherAgent(session_id=session_id)
        logger.info(f"创建新的智能Agent: session_id={session_id}")
    return _agents[session_id]


# ==================== 页面路由 ====================

@app.route("/")
@app.route("/teacher")
def teacher_page():
    """教师备课页面"""
    return render_template("teacher.html")


@app.route("/class_data_manager")
def class_data_manager_page():
    """班级数据管理页面"""
    return render_template("class_data_manager.html")


# ==================== API路由 ====================

@app.route("/api/teacher/plan", methods=["POST"])
def teacher_plan():
    """
    智能方案生成API（非流式）

    智能特性：
    - 深度理解用户意图
    - 动态策略选择
    - 自适应执行

    请求体:
        {
            "user_text": "用户输入",
            "session_id": "会话ID（可选）"
        }

    返回:
        {
            "response": "Agent回复",
            "session_info": {...}
        }
    """
    try:
        data = request.get_json() or {}
        # 兼容前端的 message 字段和 user_text 字段
        user_text = data.get("user_text", "") or data.get("message", "")
        user_text = user_text.strip()
        session_id = data.get("session_id", "default")

        if not user_text:
            return jsonify({"error": "user_text不能为空"}), 400

        agent = get_agent(session_id)
        logger.info(f"[{session_id}] 用户输入: {user_text[:50]}...")

        response = agent.chat(user_text)

        return jsonify({
            "response": response,
            "session_info": agent.get_session_info()
        })

    except Exception as e:
        logger.error(f"方案生成失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/teacher/plan/stream", methods=["POST"])
def teacher_plan_stream():
    """
    智能方案生成API（流式）

    智能特性：
    - 实时思考过程展示
    - 动态调整生成策略
    - 流畅的用户体验

    请求体:
        {
            "user_text": "用户输入",
            "session_id": "会话ID（可选）"
        }

    返回:
        SSE流式响应
    """
    try:
        data = request.get_json() or {}
        # 兼容前端的 message 字段和 user_text 字段
        user_text = data.get("user_text", "") or data.get("message", "")
        user_text = user_text.strip()
        session_id = data.get("session_id", "default")

        if not user_text:
            return jsonify({"error": "user_text不能为空"}), 400

        agent = get_agent(session_id)
        logger.info(f"[{session_id}] 流式请求: {user_text[:50]}...")

        def generate():
            try:
                for chunk in agent.chat_stream(user_text):
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"流式生成失败: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"流式方案生成失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/teacher/reset", methods=["POST"])
def teacher_reset():
    """重置会话"""
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id", "default")

        if session_id in _agents:
            _agents[session_id].reset()
            del _agents[session_id]

        return jsonify({"success": True, "message": "会话已重置"})

    except Exception as e:
        logger.error(f"重置失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/teacher/session", methods=["GET"])
def get_session_info():
    """获取会话信息"""
    try:
        session_id = request.args.get("session_id", "default")

        if session_id in _agents:
            return jsonify({
                "success": True,
                "data": _agents[session_id].get_session_info()
            })
        else:
            return jsonify({
                "success": True,
                "data": {"session_id": session_id, "message": "会话不存在"}
            })

    except Exception as e:
        logger.error(f"获取会话信息失败: {e}")
        return jsonify({"error": str(e)}), 500


# ==================== 班级数据管理API ====================
# 复用原项目的analyze_class_data模块

# 导入原项目的班级数据分析模块
sys.path.insert(0, str(ORIGINAL_PROJECT_DIR))
from analyze_class_data import (
    analyze_uploaded_file,
    analyze_class_file,
    analyze_with_llm,
    update_class_profile,
    delete_class_profile,
    get_all_class_profiles,
)
import pandas as pd
from io import BytesIO


@app.route('/api/class_data/upload', methods=['POST'])
def upload_class_data():
    """上传班级体测数据并分析（非流式）"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '请上传文件'}), 400

        file = request.files['file']
        class_name = request.form.get('class_name', '').strip()

        if not class_name:
            return jsonify({'success': False, 'message': '请提供班级名称'}), 400

        if file.filename == '':
            return jsonify({'success': False, 'message': '请选择文件'}), 400

        file_content = file.read()
        result = analyze_uploaded_file(file_content, class_name)

        if result.get('success'):
            return jsonify({'success': True, 'data': result})
        else:
            return jsonify({'success': False, 'message': result.get('error', '分析失败')}), 500

    except Exception as e:
        return jsonify({'success': False, 'message': f'分析失败: {str(e)}'}), 500


@app.route('/api/class_data/upload_stream', methods=['POST'])
def upload_class_data_stream():
    """上传班级体测数据并流式分析"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '请上传文件'}), 400

        file = request.files['file']
        class_name = request.form.get('class_name', '').strip()

        if not class_name:
            return jsonify({'success': False, 'message': '请提供班级名称'}), 400

        if file.filename == '':
            return jsonify({'success': False, 'message': '请选择文件'}), 400

        file_content = file.read()

        def generate():
            try:
                import io
                df = pd.read_excel(io.BytesIO(file_content))
                profile = None

                for chunk in analyze_with_llm(df, class_name):
                    if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "__PROFILE__":
                        profile = chunk[1]
                    elif isinstance(chunk, str):
                        yield f"data: {json.dumps({'type': 'progress', 'content': chunk}, ensure_ascii=False)}\n\n"

                if profile:
                    update_class_profile(class_name, profile)
                    yield f"data: {json.dumps({'type': 'success', 'profile': profile, 'message': '✅ 分析完成并已保存！'}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': '❌ 分析失败：未获取到分析结果'}, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': f'❌ 分析失败: {str(e)}'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'success': False, 'message': f'分析失败: {str(e)}'}), 500


@app.route('/api/class_data/profiles', methods=['GET'])
def get_class_profiles_api():
    """获取所有班级配置"""
    try:
        profiles = get_all_class_profiles()
        return jsonify({"success": True, "data": profiles, "count": len(profiles)})
    except Exception as e:
        return jsonify({"success": False, "message": f"获取班级配置失败: {str(e)}"}), 500


@app.route('/api/class_data/profile/<class_name>', methods=['DELETE'])
def delete_class_profile_api(class_name):
    """删除班级配置"""
    try:
        success = delete_class_profile(class_name)
        if success:
            return jsonify({"success": True, "message": f"已删除班级配置: {class_name}"})
        else:
            return jsonify({"success": False, "message": f"班级配置不存在: {class_name}"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": f"删除失败: {str(e)}"}), 500


@app.route('/api/class_data/download/<class_name>', methods=['GET'])
def download_class_excel(class_name):
    """下载班级配置的Excel文件"""
    try:
        profiles = get_all_class_profiles()
        if class_name not in profiles:
            return jsonify({"success": False, "message": f"班级配置不存在: {class_name}"}), 404

        profile = profiles[class_name]
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: 班级基本信息
            basic_info = {
                '班级名称': [class_name],
                '年级': [f"{profile.get('grades_query', '')}年级"],
                '薄弱项': [', '.join(profile.get('weaknesses', []))],
                '描述': [profile.get('description', '')]
            }
            df_basic = pd.DataFrame(basic_info)
            df_basic.to_excel(writer, sheet_name='班级信息', index=False)

            # Sheet 2: 学生分组详情
            if 'student_groups' in profile and profile['student_groups']:
                all_students = []
                for group_key, group_info in profile['student_groups'].items():
                    weakness_items = ', '.join(group_info.get('weakness_items', []))
                    if 'student_details' in group_info and group_info['student_details']:
                        for student in group_info['student_details']:
                            student_id = student.get('学生编号', '') or student.get('学号', '') or student.get('编号', '')
                            student_name = student.get('姓名', '')
                            student_index = student.get('序号', '')
                            if not student_name and student_index:
                                student_name = f'学生{student_index}'
                            student_row = {
                                '分组': group_key,
                                '薄弱项目': weakness_items,
                                '学生编号': str(student_id) if student_id else '',
                                '姓名': student_name
                            }
                            all_students.append(student_row)

                if all_students:
                    df_students = pd.DataFrame(all_students)
                    df_students.to_excel(writer, sheet_name='学生分组', index=False)

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={'Content-Disposition': f'attachment; filename={class_name}.xlsx'}
        )

    except Exception as e:
        return jsonify({"success": False, "message": f"下载失败: {str(e)}"}), 500


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent版本 - 体育教师备课助手")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5001, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    print(f"🚀 Agent版本启动中...")
    print(f"📍 访问地址: http://{args.host}:{args.port}")
    print(f"📍 教师备课: http://{args.host}:{args.port}/teacher")
    print(f"📍 班级管理: http://{args.host}:{args.port}/class_data_manager")

    app.run(host=args.host, port=args.port, debug=args.debug)

