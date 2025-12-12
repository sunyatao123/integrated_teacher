"""
体测数据分析模块：解析班级体测数据，生成和更新class_profiles.json
提供API接口供Flask应用调用
使用大模型进行智能分析
"""
import pandas as pd
import json
import os
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import io

# 配置日志
def setup_analyzer_logger():
    """配置分析器日志系统"""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("analyzer")
    logger.setLevel(logging.DEBUG if os.getenv('DEBUG_AI', '1') == '1' else logging.INFO)

    if logger.handlers:
        return logger

    log_file = log_dir / "analyzer.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_analyzer_logger()

# 年级编号到年级名称的映射（已废弃，改为从班级名称提取）
GRADE_MAPPING = {
    14: "1", 15: "2", 16: "3", 17: "4", 18: "5",
    19: "6", 20: "7", 21: "8", 22: "9"
}

def extract_grade_from_class_name(class_name: str) -> str:
    """
    从班级名称中提取年级

    支持的格式：
    - "五年级1班" → "5"
    - "一年级1班" → "1"
    - "3年级2班" → "3"
    - "九年级1班" → "9"

    参数:
        class_name: 班级名称

    返回:
        年级字符串（如"1"、"5"），如果提取失败返回"1"
    """
    import re

    # 中文数字到阿拉伯数字的映射
    cn_num_map = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9'
    }

    # 先将中文数字转换为阿拉伯数字
    normalized_name = class_name
    for cn, num in cn_num_map.items():
        normalized_name = normalized_name.replace(cn, num)

    # 使用正则表达式提取年级
    # 匹配模式：数字 + "年级"
    match = re.search(r'(\d+)年级', normalized_name)
    if match:
        grade = match.group(1)
        logger.debug(f"从班级名称 '{class_name}' 中提取到年级: {grade}")
        return grade

    # 如果没有匹配到，返回默认值
    logger.warning(f"无法从班级名称 '{class_name}' 中提取年级，使用默认值 '1'")
    return "1"

# 数据库规定的6个薄弱维度
ALLOWED_WEAKNESSES = ["形态", "耐力", "力量", "柔韧", "速度", "机能"]

# 体测项目到薄弱维度的映射（只能使用数据库规定的6个维度）
# 注意：只关注体测数据中的"等级"列，如"体重等级"、"50米跑等级"等
# 这里的项目名称必须与Excel中的列名前缀完全一致（去掉"等级"后缀）
WEAKNESS_MAPPING = {
    "50米跑": "速度",
    "一分钟仰卧起坐": "力量",
    "引体向上": "力量",
    "坐位体前屈": "柔韧",
    "一分钟跳绳": "机能",
    "立定跳远": "力量",
    "800米跑": "耐力",
    "1000米跑": "耐力",
    "50米×8往返跑": "耐力",
    "肺活量": "机能",
    "体重": "形态"
    # 注意：不包含"身高"和"BMI"，因为标准体测数据中没有"身高等级"和"BMI等级"列
}

def analyze_student_weaknesses(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    分析每个学生的薄弱项

    参数:
        df: 体测数据DataFrame

    返回:
        字典，key为学生姓名，value为薄弱维度列表
    """
    student_weaknesses = {}

    # 遍历每个学生
    for idx, row in df.iterrows():
        # 优先使用学生编号作为标识，其次是姓名，最后才是序号
        # 尝试从多个可能的列获取学生编号
        student_id = row.get('学生编号', '') or row.get('学号', '') or row.get('编号', '')
        student_name = row.get('姓名', '')

        if student_id:
            student_key = str(student_id)  # 使用学生编号作为主键
        elif student_name:
            student_key = student_name  # 使用姓名作为主键
        else:
            student_key = f'学生{idx+1}'  # 最后才使用序号

        weaknesses = set()  # 使用集合避免重复

        # 检查每个体测项目（使用全局WEAKNESS_MAPPING）
        for item, dimension in WEAKNESS_MAPPING.items():
            grade_col = f"{item}等级"
            if grade_col in df.columns:
                grade = row.get(grade_col)

                # 体重等级使用特殊的分类系统
                if item == "体重":
                    # 体重等级：正常、超重、肥胖、低体重
                    # 只有"正常"才不是薄弱项
                    if grade in ["超重", "肥胖", "低体重"]:
                        weaknesses.add(dimension)
                else:
                    # 其他项目使用标准分类：优秀、良好、及格、不及格
                    # 如果成绩为"不及格"或"及格"，则该维度为薄弱项
                    if grade in ["不及格", "及格"]:
                        weaknesses.add(dimension)

        # 只保存有薄弱项的学生
        if weaknesses:
            student_weaknesses[student_key] = sorted(list(weaknesses))

    return student_weaknesses


def group_students_by_weakness(student_weaknesses: Dict[str, List[str]], df: pd.DataFrame, class_weaknesses: List[str] = None) -> Dict[str, Dict]:
    """
    按班级薄弱项对学生进行分组

    新逻辑：
    1. 只针对班级的2-3个薄弱项进行分组
    2. 每个学生只看这2-3个维度的表现
    3. 分组结果如：力量薄弱组、速度薄弱组、力量+速度薄弱组等

    参数:
        student_weaknesses: 学生薄弱项字典（来自analyze_student_weaknesses）
        df: 体测数据DataFrame（用于获取学生详细信息和体测项目信息）
        class_weaknesses: 班级的薄弱项列表（如["力量", "速度"]），如果为None则使用所有薄弱项

    返回:
        分组信息字典，key为薄弱项组合（如"力量"、"速度"、"力量+速度"），value为该组的详细信息
    """
    groups = {}

    # 如果没有指定班级薄弱项，使用所有薄弱项（兼容旧逻辑）
    if class_weaknesses is None:
        class_weaknesses = list(set([w for weaknesses in student_weaknesses.values() for w in weaknesses]))

    # 遍历每个学生，只看班级薄弱项的表现
    for idx, row in df.iterrows():
        # 优先使用学生编号作为标识，其次是姓名，最后才是序号
        # 尝试从多个可能的列获取学生编号
        student_id = row.get('学生编号', '') or row.get('学号', '') or row.get('编号', '')
        student_name = row.get('姓名', '')

        if student_id:
            student_key = str(student_id)  # 使用学生编号作为主键
        elif student_name:
            student_key = student_name  # 使用姓名作为主键
        else:
            student_key = f'学生{idx+1}'  # 最后才使用序号

        # 获取学生的详细信息
        student_info = {
            "序号": idx + 1,  # 添加序号字段，从1开始
            "学生编号": str(student_id) if student_id else '',
            "姓名": student_name if student_name else ''  # 如果没有姓名，保持为空
        }

        # 尝试获取学号（如果存在，与学生编号不同）
        if '学号' in df.columns:
            student_info["学号"] = str(row.get('学号', '')) if row.get('学号', '') else ''

        # 尝试获取其他可能的学生信息字段
        for col in ['班级', '性别', '年龄']:
            if col in df.columns:
                val = row.get(col, '')
                student_info[col] = str(val) if val else ''

        # 只检查班级薄弱项对应的维度
        student_class_weaknesses = []
        if student_key in student_weaknesses:
            # 获取该学生的所有薄弱项
            all_weaknesses = student_weaknesses[student_key]
            # 只保留属于班级薄弱项的部分
            student_class_weaknesses = [w for w in all_weaknesses if w in class_weaknesses]

        # 如果该学生在班级薄弱项上没有问题，跳过
        if not student_class_weaknesses:
            continue

        # 生成分组key（如"力量"、"力量+速度"）
        group_key = "+".join(sorted(student_class_weaknesses))

        if group_key not in groups:
            groups[group_key] = {
                "count": 0,
                "students": [],
                "student_details": [],  # 新增：学生详细信息列表
                "weakness_items": []
            }

        groups[group_key]["count"] += 1
        groups[group_key]["students"].append(student_key)
        groups[group_key]["student_details"].append(student_info)

    # 为每个分组找出对应的体测项目（使用全局WEAKNESS_MAPPING）
    for group_key, group_info in groups.items():
        weakness_dims = group_key.split("+")
        items = []
        for item, dimension in WEAKNESS_MAPPING.items():
            if dimension in weakness_dims:
                grade_col = f"{item}等级"
                if grade_col in df.columns:
                    items.append(item)
        # 去重
        group_info["weakness_items"] = list(set(items))

    # 按人数降序排序
    sorted_groups = dict(sorted(groups.items(), key=lambda x: x[1]["count"], reverse=True))

    return sorted_groups


def analyze_class_weakness(df: pd.DataFrame, class_name: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    分析班级的薄弱项

    返回: (薄弱项列表, 薄弱项详细信息字典, 薄弱项对应的体测项目字典)
    """
    weaknesses = []
    weakness_details = {}
    weakness_items = {}  # 记录每个薄弱维度对应的体测项目

    weakness_scores = {}

    # 分析各项体测数据的等级分布（使用全局WEAKNESS_MAPPING）
    for item, dimension in WEAKNESS_MAPPING.items():
        grade_col = f"{item}等级"
        if grade_col not in df.columns:
            continue

        # 统计等级分布
        grade_counts = df[grade_col].value_counts()
        total = len(df[df[grade_col].notna()])

        if total == 0:
            continue

        # 体重等级使用特殊的分类系统
        if item == "体重":
            # 体重等级：正常、超重、肥胖、低体重
            normal_count = grade_counts.get("正常", 0)
            overweight_count = grade_counts.get("超重", 0)
            obese_count = grade_counts.get("肥胖", 0)
            underweight_count = grade_counts.get("低体重", 0)

            normal_rate = normal_count / total * 100
            overweight_rate = overweight_count / total * 100
            obese_rate = obese_count / total * 100
            underweight_rate = underweight_count / total * 100

            # 计算薄弱分数（正常率越低，分数越高表示越薄弱）
            weakness_score = (100 - normal_rate) + obese_rate * 2 + overweight_rate * 1.5 + underweight_rate * 1.5

            # 为了统一接口，将体重等级映射到标准等级
            excellent_count = normal_count
            good_count = 0
            pass_count = overweight_count + underweight_count
            fail_count = obese_count

            excellent_rate = normal_rate
            good_rate = 0
            pass_rate = overweight_rate + underweight_rate
            fail_rate = obese_rate
        else:
            # 其他项目使用标准分类：优秀、良好、及格、不及格
            excellent_count = grade_counts.get("优秀", 0)
            good_count = grade_counts.get("良好", 0)
            pass_count = grade_counts.get("及格", 0)
            fail_count = grade_counts.get("不及格", 0)

            excellent_rate = excellent_count / total * 100
            good_rate = good_count / total * 100
            pass_rate = pass_count / total * 100
            fail_rate = fail_count / total * 100

            # 计算薄弱分数（优秀率越低、及格率越高，分数越高表示越薄弱）
            weakness_score = (100 - excellent_rate) + pass_rate + fail_rate * 2

        # 修复：对于同一维度的多个项目，选择最薄弱的那个
        if dimension not in weakness_scores or weakness_score > weakness_scores[dimension]["score"]:
            weakness_scores[dimension] = {
                "score": weakness_score,
                "item": item,
                "excellent_rate": excellent_rate,
                "good_rate": good_rate,
                "pass_rate": pass_rate,
                "fail_rate": fail_rate,
                "excellent_count": excellent_count,
                "good_count": good_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "total": total
            }

    # 找出最薄弱的2个维度
    sorted_weaknesses = sorted(weakness_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    for dimension, stats in sorted_weaknesses[:2]:
        weaknesses.append(dimension)
        weakness_items[dimension] = stats['item']  # 记录对应的体测项目

        # 生成详细描述
        detail = f"从体测数据来看，{dimension}是{class_name}的薄弱项：{stats['item']}"

        if stats['excellent_count'] == 0:
            detail += f"无'优秀'等级学生，"
        else:
            detail += f"仅{stats['excellent_count']}人（占比{stats['excellent_rate']:.1f}%）达到'优秀'，"

        if stats['good_count'] > 0:
            detail += f"{stats['good_count']}人（占比{stats['good_rate']:.1f}%）达到'良好'，"

        detail += f"{stats['pass_count']}人（占比{stats['pass_rate']:.1f}%）为'及格'"

        if stats['fail_count'] > 0:
            detail += f"，{stats['fail_count']}人（占比{stats['fail_rate']:.1f}%）为'不及格'"

        detail += f"，{dimension}素质提升需求迫切。"

        weakness_details[dimension] = detail

    return weaknesses, weakness_details, weakness_items


def analyze_class_file(file_path: Path) -> Dict:
    """分析单个班级文件"""
    df = pd.read_excel(file_path)
    class_name = file_path.stem  # 例如：一年级1班

    # 从班级名称中提取年级（而不是从Excel文档内部的"年级编号"列）
    grade_query = extract_grade_from_class_name(class_name)

    # 分析班级整体薄弱项
    weaknesses, weakness_details, weakness_test_items = analyze_class_weakness(df, class_name)

    # 分析学生个体薄弱项
    student_weaknesses = analyze_student_weaknesses(df)

    # 按班级薄弱项分组（只针对班级的2-3个薄弱项）
    student_groups = group_students_by_weakness(student_weaknesses, df, class_weaknesses=weaknesses)

    # 构建班级配置
    # 生成简洁的描述
    weakness_desc_items = []
    for weakness in weaknesses:
        test_item = weakness_test_items.get(weakness, "")
        if test_item:
            weakness_desc_items.append(f"{weakness}（{test_item}）")
        else:
            weakness_desc_items.append(weakness)

    description = f"{class_name}体质监测核心薄弱维度：" + "、".join(weakness_desc_items) if weakness_desc_items else f"{class_name}体质监测数据"

    profile = {
        "grades_query": grade_query,
        "trained_weaknesses": "、".join(weaknesses) if weaknesses else "",
        "count_query": "",
        "semantic_query": "",
        "description": description,
        "weakness_details": weakness_details,
        "student_groups": student_groups  # 新增：学生分组信息
    }

    return class_name, profile


def generate_class_profiles(class_data_dir="class_data", output_file="prompts/class_profiles.json", max_classes=10):
    """
    生成class_profiles.json文件
    
    参数:
        class_data_dir: 班级数据文件夹
        output_file: 输出JSON文件路径
        max_classes: 最多处理多少个班级（用于测试，设为None处理全部）
    """
    class_data_path = Path(class_data_dir)
    profiles = {}
    
    # 获取所有班级文件
    class_files = sorted(class_data_path.glob("*.xlsx"))
    
    if max_classes:
        class_files = class_files[:max_classes]
    
    logger.info(f"开始分析 {len(class_files)} 个班级...")
    
    for idx, file_path in enumerate(class_files, 1):
        try:
            class_name, profile = analyze_class_file(file_path)
            profiles[class_name] = profile
            logger.info(f"[{idx}/{len(class_files)}] 分析完成: {class_name}")
        except Exception as e:
            logger.error(f"[{idx}/{len(class_files)}] 分析失败: {file_path.name}, 错误: {e}")
    
    # 保存到JSON文件
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n生成完成！共分析 {len(profiles)} 个班级，保存到 {output_file}")
    return profiles


def analyze_with_llm(df: pd.DataFrame, class_name: str) -> Generator[str, None, Dict]:
    """
    使用大模型分析体测数据（流式输出）

    参数:
        df: 体测数据DataFrame
        class_name: 班级名称

    返回:
        生成器，yield分析过程，最后返回分析结果
    """
    from ai_model_optimized import OptimizedAIModel

    try:
        # 从班级名称中提取年级（而不是从Excel文档内部的"年级编号"列）
        grade_query = extract_grade_from_class_name(class_name)

        yield f"📊 开始分析 {class_name} 的体测数据...\n\n"
        yield f"✅ 检测到年级：{grade_query}年级\n"
        yield f"✅ 学生人数：{len(df)}人\n\n"

        # 统计各项体测数据
        yield "📈 正在统计各项体测指标...\n\n"

        # 使用全局WEAKNESS_MAPPING统计所有项目
        stats_text = ""
        for item, dimension in WEAKNESS_MAPPING.items():
            grade_col = f"{item}等级"
            if grade_col in df.columns:
                grade_counts = df[grade_col].value_counts()
                total = len(df[df[grade_col].notna()])
                if total > 0:
                    # 体重等级使用特殊的分类系统
                    if item == "体重":
                        stats_text += f"- {item}（{dimension}）：正常{grade_counts.get('正常', 0)}人，超重{grade_counts.get('超重', 0)}人，肥胖{grade_counts.get('肥胖', 0)}人，低体重{grade_counts.get('低体重', 0)}人\n"
                    else:
                        stats_text += f"- {item}（{dimension}）：优秀{grade_counts.get('优秀', 0)}人，良好{grade_counts.get('良好', 0)}人，及格{grade_counts.get('及格', 0)}人，不及格{grade_counts.get('不及格', 0)}人\n"

        yield stats_text + "\n"

        # 调用大模型分析
        yield "🤖 正在使用AI分析薄弱项...\n\n"

        model = OptimizedAIModel()

        prompt = f"""你是一位专业的体育教师，请分析以下班级的体测数据，识别薄弱项。

班级：{class_name}
年级：{grade_query}年级
学生人数：{len(df)}人

各项体测数据统计：
{stats_text}

**重要规则：**
1. 薄弱项只能从以下6个维度中选择：形态、耐力、力量、柔韧、速度、机能
2. 请选择最薄弱的1-2个维度
3. 对每个薄弱维度，给出详细的分析说明

请以JSON格式返回分析结果：
```json
{{
    "weaknesses": ["维度1", "维度2"],
    "weakness_details": {{
        "维度1": "详细分析说明...",
        "维度2": "详细分析说明..."
    }}
}}
```"""

        messages = [
            {"role": "system", "content": "你是一位专业的体育教师，擅长分析学生体测数据。"},
            {"role": "user", "content": prompt}
        ]

        try:
            # 记录模型调用message
            logger.info(f"[MODEL_CALL] 班级数据分析 - message:")
            logger.info(f"  模型: {model.model}")
            logger.info(f"  Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
            logger.info(f"  参数: max_tokens=1000, temperature=0.3")
            start_time = time.time()
            
            response = model.client.chat.completions.create(
                model=model.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            elapsed_time = time.time() - start_time
            response_text = response.choices[0].message.content.strip()
            
            # 记录模型调用Response
            logger.info(f"[MODEL_CALL] 班级数据分析 - Response:")
            logger.info(f"  响应时间: {elapsed_time:.3f}秒 ({elapsed_time*1000:.1f}毫秒)")
            logger.info(f"  响应内容长度: {len(response_text)} 字符")
            logger.info(f"  响应内容: {repr(response_text)}")
            
            yield f"AI分析结果：\n{response_text}\n\n"
        except Exception as api_error:
            # 记录详细的API错误信息
            logger.error(f"AI模型API调用失败: {api_error}")
            yield f"⚠️ AI分析失败（{str(api_error)}），使用传统方法分析...\n\n"
            # 使用传统方法分析
            weaknesses, weakness_details, _ = analyze_class_weakness(df, class_name)
            weaknesses = [w for w in weaknesses if w in ALLOWED_WEAKNESSES][:2]

            yield f"✅ 识别到薄弱项：{', '.join(weaknesses)}\n\n"

            # 分析学生个体薄弱项和分组
            yield "👥 正在分析学生个体薄弱项...\n"
            student_weaknesses = analyze_student_weaknesses(df)
            yield f"✅ 已分析 {len(student_weaknesses)} 名学生的薄弱项\n\n"

            yield f"📊 正在按班级薄弱项（{', '.join(weaknesses)}）对学生分组...\n"
            student_groups = group_students_by_weakness(student_weaknesses, df, class_weaknesses=weaknesses)
            yield f"✅ 已生成 {len(student_groups)} 个学生分组\n\n"

            # 构建描述
            description = f"{class_name}体质监测核心薄弱维度：" + "、".join(weaknesses) if weaknesses else f"{class_name}体质监测数据"

            profile = {
                "grades_query": grade_query,
                "trained_weaknesses": "、".join(weaknesses) if weaknesses else "",
                "count_query": "",
                "semantic_query": "",
                "description": description,
                "weakness_details": weakness_details,
                "student_groups": student_groups
            }

            yield "💾 正在保存配置...\n"
            yield ("__PROFILE__", profile)
            return

        # 解析JSON结果
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            analysis_result = json.loads(json_match.group(1))
            weaknesses = analysis_result.get('weaknesses', [])
            weakness_details = analysis_result.get('weakness_details', {})
        else:
            # 如果没有找到JSON，使用传统方法分析
            yield "⚠️ AI返回格式异常，使用传统方法分析...\n\n"
            weaknesses, weakness_details, _ = analyze_class_weakness(df, class_name)

        # 确保薄弱项在允许的范围内
        weaknesses = [w for w in weaknesses if w in ALLOWED_WEAKNESSES][:2]

        yield f"✅ 识别到薄弱项：{', '.join(weaknesses)}\n\n"

        # 分析学生个体薄弱项和分组
        yield "👥 正在分析学生个体薄弱项...\n"
        student_weaknesses = analyze_student_weaknesses(df)
        yield f"✅ 已分析 {len(student_weaknesses)} 名学生的薄弱项\n\n"

        yield f"📊 正在按班级薄弱项（{', '.join(weaknesses)}）对学生分组...\n"
        student_groups = group_students_by_weakness(student_weaknesses, df, class_weaknesses=weaknesses)
        yield f"✅ 已生成 {len(student_groups)} 个学生分组\n\n"

        # 构建描述
        description = f"{class_name}体质监测核心薄弱维度：" + "、".join(weaknesses) if weaknesses else f"{class_name}体质监测数据"

        profile = {
            "grades_query": grade_query,
            "trained_weaknesses": "、".join(weaknesses) if weaknesses else "",
            "count_query": "",
            "semantic_query": "",
            "description": description,
            "weakness_details": weakness_details,
            "student_groups": student_groups
        }

        yield "💾 正在保存配置...\n"

        # 使用特殊标记来标识这是最终结果
        yield ("__PROFILE__", profile)

    except Exception as e:
        logger.error(f"分析失败: {e}", exc_info=True)
        yield f"❌ 分析失败：{str(e)}\n"
        raise e


def analyze_uploaded_file(file_content: bytes, class_name: str, output_file: str = "prompts/class_profiles.json") -> Dict:
    """
    分析上传的体测数据文件

    参数:
        file_content: 文件内容（字节）
        class_name: 班级名称
        output_file: 输出JSON文件路径

    返回:
        分析结果字典
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(io.BytesIO(file_content))

        # 从班级名称中提取年级（而不是从Excel文档内部的"年级编号"列）
        grade_query = extract_grade_from_class_name(class_name)

        # 分析班级整体薄弱项
        weaknesses, weakness_details, weakness_test_items = analyze_class_weakness(df, class_name)

        # 分析学生个体薄弱项
        student_weaknesses = analyze_student_weaknesses(df)

        # 按班级薄弱项分组（只针对班级的2-3个薄弱项）
        student_groups = group_students_by_weakness(student_weaknesses, df, class_weaknesses=weaknesses)

        # 构建班级配置
        weakness_desc_items = []
        for weakness in weaknesses:
            test_item = weakness_test_items.get(weakness, "")
            if test_item:
                weakness_desc_items.append(f"{weakness}（{test_item}）")
            else:
                weakness_desc_items.append(weakness)

        description = f"{class_name}体质监测核心薄弱维度：" + "、".join(weakness_desc_items) if weakness_desc_items else f"{class_name}体质监测数据"

        profile = {
            "grades_query": grade_query,
            "trained_weaknesses": "、".join(weaknesses) if weaknesses else "",
            "count_query": "",
            "semantic_query": "",
            "description": description,
            "weakness_details": weakness_details,
            "student_groups": student_groups  # 新增：学生分组信息
        }

        # 更新JSON文件
        update_class_profile(class_name, profile, output_file)

        return {
            "success": True,
            "class_name": class_name,
            "profile": profile
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def update_class_profile(class_name: str, profile: Dict, output_file: str = "prompts/class_profiles.json"):
    """
    更新class_profiles.json文件

    参数:
        class_name: 班级名称
        profile: 班级配置
        output_file: 输出JSON文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    # 读取现有配置
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    profiles = json.loads(content)
                else:
                    profiles = {}
        except (json.JSONDecodeError, ValueError):
            # 如果JSON文件损坏或为空，初始化为空字典
            profiles = {}
    else:
        profiles = {}

    # 更新配置
    profiles[class_name] = profile

    # 保存到JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def delete_class_profile(class_name: str, output_file: str = "prompts/class_profiles.json") -> bool:
    """
    删除班级配置

    参数:
        class_name: 班级名称
        output_file: 输出JSON文件路径

    返回:
        是否删除成功
    """
    output_path = Path(output_file)

    if not output_path.exists():
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                profiles = json.loads(content)
            else:
                profiles = {}
    except (json.JSONDecodeError, ValueError):
        # 如果JSON文件损坏或为空，返回False
        return False

    if class_name in profiles:
        del profiles[class_name]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

        return True

    return False


def get_all_class_profiles(output_file: str = "prompts/class_profiles.json") -> Dict:
    """
    获取所有班级配置

    参数:
        output_file: JSON文件路径

    返回:
        所有班级配置字典
    """
    output_path = Path(output_file)

    if not output_path.exists():
        return {}

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            else:
                return {}
    except (json.JSONDecodeError, ValueError):
        # 如果JSON文件损坏或为空，返回空字典
        return {}


if __name__ == "__main__":
    # 测试：分析class_data文件夹中的所有班级
    generate_class_profiles(max_classes=None)

