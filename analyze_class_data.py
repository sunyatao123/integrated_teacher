"""
体测数据分析模块：解析班级体测数据，生成和更新class_profiles.json
提供API接口供Flask应用调用
使用大模型进行智能分析
"""
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import io

# 年级编号到年级名称的映射
GRADE_MAPPING = {
    14: "1", 15: "2", 16: "3", 17: "4", 18: "5",
    19: "6", 20: "7", 21: "8", 22: "9"
}

# 数据库规定的6个薄弱维度
ALLOWED_WEAKNESSES = ["形态", "耐力", "力量", "柔韧", "速度", "机能"]

# 体测项目到薄弱维度的映射（只能使用数据库规定的6个维度）
WEAKNESS_MAPPING = {
    "50米跑": "速度",
    "一分钟仰卧起坐": "力量",
    "引体向上": "力量",
    "坐位体前屈": "柔韧",
    "一分钟跳绳": "速度",  # 跳绳归入速度
    "立定跳远": "力量",    # 爆发力归入力量
    "800米跑": "耐力",
    "1000米跑": "耐力",
    "肺活量": "机能",      # 心肺功能归入机能
    "身高": "形态",
    "体重": "形态",
    "BMI": "形态"
}

def analyze_class_weakness(df: pd.DataFrame, class_name: str) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    分析班级的薄弱项

    返回: (薄弱项列表, 薄弱项详细信息字典, 薄弱项对应的体测项目字典)
    """
    weaknesses = []
    weakness_details = {}
    weakness_items = {}  # 记录每个薄弱维度对应的体测项目

    # 分析各项体测数据的等级分布（只使用数据库规定的6个维度）
    test_items = {
        "50米跑": "速度",
        "一分钟仰卧起坐": "力量",
        "坐位体前屈": "柔韧",
        "一分钟跳绳": "速度",
        "立定跳远": "力量",
        "800米跑": "耐力",
        "1000米跑": "耐力",
        "肺活量": "机能",
        "身高": "形态",
        "体重": "形态"
    }

    weakness_scores = {}
    
    for item, dimension in test_items.items():
        grade_col = f"{item}等级"
        if grade_col not in df.columns:
            continue
            
        # 统计等级分布
        grade_counts = df[grade_col].value_counts()
        total = len(df[df[grade_col].notna()])
        
        if total == 0:
            continue
        
        # 计算优秀率和及格率
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

    # 获取年级编号
    grade_code = df['年级编号'].iloc[0] if len(df) > 0 else 14
    grade_query = GRADE_MAPPING.get(grade_code, "1")

    # 分析薄弱项
    weaknesses, weakness_details, weakness_test_items = analyze_class_weakness(df, class_name)

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
        "weakness_details": weakness_details
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
    
    print(f"开始分析 {len(class_files)} 个班级...")
    
    for idx, file_path in enumerate(class_files, 1):
        try:
            class_name, profile = analyze_class_file(file_path)
            profiles[class_name] = profile
            print(f"[{idx}/{len(class_files)}] 分析完成: {class_name}")
        except Exception as e:
            print(f"[{idx}/{len(class_files)}] 分析失败: {file_path.name}, 错误: {e}")
    
    # 保存到JSON文件
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！共分析 {len(profiles)} 个班级，保存到 {output_file}")
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
        # 获取年级编号
        grade_code = df['年级编号'].iloc[0] if len(df) > 0 else 14
        grade_query = GRADE_MAPPING.get(grade_code, "1")

        yield f"📊 开始分析 {class_name} 的体测数据...\n\n"
        yield f"✅ 检测到年级：{grade_query}年级\n"
        yield f"✅ 学生人数：{len(df)}人\n\n"

        # 统计各项体测数据
        yield "📈 正在统计各项体测指标...\n\n"

        test_items = {
            "50米跑": "速度",
            "一分钟仰卧起坐": "力量",
            "坐位体前屈": "柔韧",
            "一分钟跳绳": "速度",
            "立定跳远": "力量",
            "800米跑": "耐力",
            "1000米跑": "耐力",
            "肺活量": "机能",
            "身高": "形态",
            "体重": "形态"
        }

        stats_text = ""
        for item, dimension in test_items.items():
            grade_col = f"{item}等级"
            if grade_col in df.columns:
                grade_counts = df[grade_col].value_counts()
                total = len(df[df[grade_col].notna()])
                if total > 0:
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

        response = model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=1000,
            temperature=0.3
        )

        response_text = response.choices[0].message.content.strip()
        yield f"AI分析结果：\n{response_text}\n\n"

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

        # 构建描述
        description = f"{class_name}体质监测核心薄弱维度：" + "、".join(weaknesses) if weaknesses else f"{class_name}体质监测数据"

        profile = {
            "grades_query": grade_query,
            "trained_weaknesses": "、".join(weaknesses) if weaknesses else "",
            "count_query": "",
            "semantic_query": "",
            "description": description,
            "weakness_details": weakness_details
        }

        yield "💾 正在保存配置...\n"

        # 使用特殊标记来标识这是最终结果
        yield ("__PROFILE__", profile)

    except Exception as e:
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

        # 获取年级编号
        grade_code = df['年级编号'].iloc[0] if len(df) > 0 else 14
        grade_query = GRADE_MAPPING.get(grade_code, "1")

        # 分析薄弱项
        weaknesses, weakness_details, weakness_test_items = analyze_class_weakness(df, class_name)

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
            "weakness_details": weakness_details
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

