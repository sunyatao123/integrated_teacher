#!/bin/bash

# ==================== 配置区域 ====================
# 可以在这里修改配置，或者通过环境变量覆盖

# 服务器配置
export HOST=${HOST:-"0.0.0.0"}        # 监听地址，0.0.0.0表示监听所有网络接口
export PORT=${PORT:-5000}             # 端口号
export DEBUG=${DEBUG:-"False"}        # 是否开启调试模式（生产环境建议False）

# 日志配置
LOG_DIR="logs"                        # 日志目录
LOG_FILE="${LOG_DIR}/app.log"         # 日志文件名
PID_FILE="${LOG_DIR}/app.pid"         # PID文件

# ==================== 启动脚本 ====================

# 创建日志目录
mkdir -p ${LOG_DIR}

# 检查是否已经在运行
if [ -f ${PID_FILE} ]; then
    PID=$(cat ${PID_FILE})
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "应用已经在运行 (PID: ${PID})"
        echo "如需重启，请先运行: ./stop.sh"
        exit 1
    else
        echo "发现旧的PID文件，但进程不存在，删除PID文件"
        rm -f ${PID_FILE}
    fi
fi

echo "=========================================="
echo "教师端AI备课助手 - 后台启动"
echo "=========================================="
echo "监听地址: ${HOST}:${PORT}"
echo "调试模式: ${DEBUG}"
echo "日志文件: ${LOG_FILE}"
echo "PID文件: ${PID_FILE}"
echo "=========================================="

# 后台启动应用
nohup python app.py > ${LOG_FILE} 2>&1 &

# 保存PID
echo $! > ${PID_FILE}

echo "应用已在后台启动 (PID: $(cat ${PID_FILE}))"
echo ""
echo "查看日志: tail -f ${LOG_FILE}"
echo "停止应用: ./stop.sh"
echo "=========================================="

