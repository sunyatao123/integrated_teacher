# 教师端AI备课助手（整合版本）

这是结合原版本和teacher版本优点的整合版本，专注于教师端功能。

## 版本特点

### 🎯 核心优势

1. **模块化架构**（来自原版本）
   - 提示词模板化管理（`prompts/` 文件夹）
   - 代码与配置分离
   - 易于维护和扩展

2. **优化的对话逻辑**（来自teacher版本）
   - 更准确的意图识别
   - 更自然的对话流程
   - 更好的方案生成质量

3. **班级管理系统**（来自原版本）
   - 支持班级配置文件（`class_profiles.json`）
   - 动态班级数据管理
   - 体测数据分析

4. **专注教师端**
   - 移除学生端功能
   - 简化代码结构
   - 提升性能

## 功能列表

### 📚 教师端备课助手

- **意图识别**：自动识别用户意图（课课练/全员运动会/闲聊）
- **参数提取**：智能提取班级、薄弱项、场地条件等参数
- **智能检索**：调用外部检索服务获取相关训练内容
- **方案生成**：基于检索结果生成个性化备课方案
- **流式输出**：支持流式和非流式两种输出模式
- **多轮对话**：支持上下文理解和多轮对话

### 📊 班级数据管理

- **数据上传**：上传班级体测Excel数据
- **智能分析**：使用AI分析班级薄弱项
- **配置管理**：查看、删除班级配置
- **批量处理**：批量分析多个班级数据

## 快速开始

### 1. 环境要求

- Python 3.8+
- pip包管理器

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，并填写配置：

```bash
# SiliconFlow API密钥（必填）
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx

# 检索服务地址（可选）
SEARCH_BASE_URL=http://127.0.0.1:8001

# 服务器端口（可选）
PORT=5000

# 调试模式（可选）
DEBUG_AI=1
```

### 4. 启动服务

**Windows用户**：
```bash
start_server.bat
```

**Linux/Mac用户**：
```bash
python app.py
```

### 5. 访问应用

- 教师端备课助手：http://127.0.0.1:5000/teacher
- 班级数据管理：http://127.0.0.1:5000/class_data_manager

## 项目结构

```
integrated_teacher/
├── app.py                          # Flask应用主文件
├── teacher_planner.py              # 教师端备课核心逻辑
├── ai_model_optimized.py           # AI模型封装
├── analyze_class_data.py           # 班级数据分析模块
├── requirements.txt                # Python依赖
├── start_server.bat                # Windows启动脚本
├── .env.example                    # 环境变量示例
├── README.md                       # 项目说明
├── prompts/                        # 提示词模板文件夹
│   ├── teacher_system_prompt.txt   # 教师助手系统提示词
│   ├── intent_recognition.txt      # 意图识别提示词
│   ├── param_extraction_system.txt # 参数提取系统提示词
│   ├── param_extraction_user.txt   # 参数提取用户提示词
│   ├── plan_generation_lesson_plan.txt      # 课课练方案生成
│   ├── plan_generation_sports_meeting.txt   # 运动会方案生成
│   ├── guidance_prompt.txt         # 引导语生成
│   └── class_profiles.json         # 班级配置文件
├── templates/                      # HTML模板
│   ├── teacher.html                # 教师端对话界面
│   └── class_data_manager.html     # 班级数据管理界面
└── class_data/                     # 班级体测数据存储
```

## API接口

### 教师端备课

- `POST /api/teacher/plan` - 非流式方案生成
- `POST /api/teacher/plan/stream` - 流式方案生成

### 班级数据管理

- `POST /api/class_data/upload` - 上传并分析班级数据（非流式）
- `POST /api/class_data/upload_stream` - 上传并分析班级数据（流式）
- `POST /api/class_data/analyze/<filename>` - 分析已有数据文件
- `GET /api/class_data/profiles` - 获取所有班级配置
- `DELETE /api/class_data/profile/<class_name>` - 删除班级配置
- `POST /api/class_data/batch_analyze` - 批量分析班级数据

## 技术栈

- **后端框架**：Flask 3.0
- **AI模型**：DeepSeek-V3.1（通过SiliconFlow API）
- **数据处理**：Pandas + OpenPyXL
- **前端**：原生HTML + JavaScript

## 许可证

本项目仅供学习和研究使用。

