@echo off
chcp 65001 >nul
echo ========================================
echo 教师端AI备课助手（整合版本）
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查虚拟环境是否存在
if not exist "venv" (
    echo [提示] 首次运行，正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo [成功] 虚拟环境创建完成
    echo.
)

REM 激活虚拟环境
echo [提示] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b 1
)

REM 检查依赖是否安装
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行，正在安装依赖...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 安装依赖失败
        pause
        exit /b 1
    )
    echo [成功] 依赖安装完成
    echo.
)

REM 检查.env文件是否存在
if not exist ".env" (
    echo [警告] 未找到.env文件，请创建.env文件并配置以下环境变量：
    echo   - SILICONFLOW_API_KEY=你的API密钥
    echo   - SEARCH_BASE_URL=检索服务地址（可选，默认http://127.0.0.1:8001）
    echo   - PORT=端口号（可选，默认5000）
    echo   - DEBUG_AI=是否开启调试（可选，默认1）
    echo.
    echo 示例.env文件内容：
    echo SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx
    echo SEARCH_BASE_URL=http://127.0.0.1:8001
    echo PORT=5000
    echo DEBUG_AI=1
    echo.
    pause
)

REM 启动服务器
echo [提示] 启动服务器...
echo.
python app.py

REM 如果服务器异常退出，暂停以便查看错误信息
if errorlevel 1 (
    echo.
    echo [错误] 服务器异常退出
    pause
)

