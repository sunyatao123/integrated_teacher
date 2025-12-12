"""
Agent 版本 Flask API 入口
"""
import os
import sys
from pathlib import Path

# 兼容直接 python app.py 运行：补充模块搜索路径
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from flask import Flask, jsonify, request, Response, render_template, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# 延迟导入，确保路径已设置
from agent_runner import (
    run_agent_dialog,
    run_agent_stream,
    get_or_create_thread_id,
    reset_session
)

app = Flask(__name__, template_folder="templates")
CORS(app)


@app.route("/")
@app.route("/agent")
def agent_page():
    """Agent 前端页面"""
    return render_template("agent_teacher.html")


@app.route("/api/agent/plan", methods=["POST"])
def agent_plan():
    """非流式 Agent API"""
    data = request.get_json() or {}
    user_text = data.get("user_text") or data.get("message") or ""
    conversation_history = data.get("conversation_history") or []
    session_id = data.get("session_id") or "default"
    
    if not user_text:
        return jsonify({"success": False, "message": "user_text不能为空"}), 400
    
    try:
        thread_id = get_or_create_thread_id(session_id)
        result = run_agent_dialog(user_text, conversation_history, thread_id)
        return jsonify({"success": True, "response": result, "session_id": session_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/agent/plan/stream", methods=["POST"])
def agent_plan_stream():
    """流式 Agent API"""
    data = request.get_json() or {}
    user_text = data.get("user_text") or data.get("message") or ""
    conversation_history = data.get("conversation_history") or []
    session_id = data.get("session_id") or "default"
    
    if not user_text:
        return jsonify({"success": False, "message": "user_text不能为空"}), 400

    @stream_with_context
    def generate():
        try:
            thread_id = get_or_create_thread_id(session_id)
            for chunk in run_agent_stream(user_text, conversation_history, thread_id):
                if chunk:  # 只输出非空内容
                    yield chunk
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"\n\n生成失败: {str(e)}"

    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用nginx缓冲
        }
    )


@app.route("/api/agent/reset", methods=["POST"])
def agent_reset():
    """重置会话"""
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    reset_session(session_id)
    return jsonify({"success": True, "message": "会话已重置"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    print(f"\n{'='*50}")
    print(f"🤖 Agent 备课助手")
    print(f"{'='*50}")
    print(f"📍 访问地址: http://localhost:{port}")
    print(f"📍 前端界面: http://localhost:{port}/agent")
    print(f"📍 调试日志: 终端实时显示")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
