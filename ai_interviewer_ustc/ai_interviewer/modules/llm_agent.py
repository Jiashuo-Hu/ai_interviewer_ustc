"""
文字入/文字出的面试对话模块，RAG 检索逻辑由 rag_engine 提供。
此前流程：用户输入 -> 直接发给 LLM；
现在流程：用户输入 -> RAG 检索 -> 组装 Prompt -> 发给 LLM。
"""
import os

from openai import OpenAI
from modules.rag_engine import get_retrieved_context
from config import DASHSCOPE_API_KEY


INTERVIEWER_STYLES = {
    "pressure_high": {
        "name": "高压面试官",
        "prompt": "保持紧迫感和压力感，问题尖锐直接，追问细节与缺陷、直击核心难点，语气简短，不给出提示。",
    },
    "friendly": {
        "name": "友好面试官",
        "prompt": "语气亲和，逐步引导，适度鼓励，若回答模糊则温和提示补充。",
    },
    "neutral": {
        "name": "标准专业面试官",
        "prompt": "保持中性、专业、条理清晰，不情绪化，专注于能力评估，正常中和的面试风格。",
    },
}
DEFAULT_STYLE = "neutral"

# 每种风格对应偏好的题目难度（用于检索过滤）
STYLE_DIFFICULTY = {
    "friendly": "easy",
    "neutral": "medium",
    "pressure_high": "hard",
}


if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def llm_stream_chat(history, user_input, domain="cs", interviewer_style=None):
    """流式对话：先检索上下文，再按风格生成提问。"""

    style_key = interviewer_style or DEFAULT_STYLE
    style_cfg = INTERVIEWER_STYLES.get(style_key, INTERVIEWER_STYLES[DEFAULT_STYLE])
    difficulty_pref = STYLE_DIFFICULTY.get(style_key, STYLE_DIFFICULTY[DEFAULT_STYLE])

    # RAG 检索上下文（按风格偏好难度过滤）
    context = get_retrieved_context(
        user_input,
        domain=domain,
        k=6,
        search_filter={"difficulty": difficulty_pref},
    )

    # 组装系统提示词
    system_prompt = f"""你是一名专业的{domain}领域面试官，风格：{style_cfg['name']}。
    风格要求：{style_cfg['prompt']}
    请根据以下专业背景知识进行提问：
    ---
    {context}
    ---
    规则：每次只提 1-2 个问题；若回答模糊则按风格要求追问；回答不确定时说明“根据现有信息无法判断”。"""

    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_input}
    ]

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            stream=True,
        )

        full_response = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield full_response

    except Exception as exc:
        yield f"抱歉，系统出现了点小故障: {str(exc)}"