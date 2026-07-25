from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import httpx
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# 阿里云百炼 DeepSeek API 配置
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = os.environ.get(
    'DEEPSEEK_API_HOST',
    'https://ws-ndrn7uqqjjn2ucra.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
)
DEEPSEEK_MODEL = "deepseek-v3"


def analyze_resume_with_llm(position: str, resume: str) -> dict:
    """使用 DeepSeek 大模型分析简历并推荐公司"""

    if not DEEPSEEK_API_KEY:
        return {
            "recommended_company": "API配置错误",
            "modification_advice": "请在 Render 环境变量中配置 DEEPSEEK_API_KEY"
        }

    print(f"🔑 API Key 长度: {len(DEEPSEEK_API_KEY)}, base_url: {DEEPSEEK_BASE_URL}")

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            http_client=httpx.Client()
        )

        prompt = f"""你是一位资深的招聘顾问和职业规划专家。请根据以下信息，为求职者提供精准的公司推荐和简历优化建议。

【目标岗位】
{position}

【求职者简历】
{resume}

请完成以下两个任务：

1. 推荐最合适的公司：基于求职者的技能、经验和目标岗位，推荐1家最匹配的公司（可以是知名互联网公司、AI公司、传统科技企业等），并简要说明推荐理由（50字以内）。

2. 简历修改建议：提供具体的、可操作的简历优化方向，包括：
   - 应该突出哪些技能和项目经验
   - 需要补充或弱化哪些内容
   - 如何调整表述以匹配目标公司和岗位的要求
   - 建议的关键词和亮点

必须严格按照以下JSON格式返回，不要有任何其他文字：
{{"recommended_company": "公司名称（推荐理由）", "modification_advice": "详细的简历修改建议，分段说明，至少200字"}}"""

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的招聘顾问，擅长分析求职者背景并提供精准的职业建议。只返回JSON格式，不要有其他文字。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()
        print(f"✅ 模型返回: {result_text[:100]}...")

        # 清理可能的 markdown 代码块
        if "```" in result_text:
            parts = result_text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    result_text = part
                    break

        result = json.loads(result_text)
        return {
            "recommended_company": result.get("recommended_company", "未知公司"),
            "modification_advice": result.get("modification_advice", "暂无建议")
        }

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {str(e)}, 原始: {result_text[:200]}")
        # 解析失败时直接返回原始文本，不让用户看到报错
        return {
            "recommended_company": "分析完成",
            "modification_advice": result_text
        }
    except Exception as e:
        print(f"❌ LLM调用失败: {str(e)}")
        return {
            "recommended_company": "分析失败",
            "modification_advice": f"系统错误：{str(e)}"
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请求格式错误'}), 400

        position = data.get('position', '').strip()
        resume = data.get('resume', '').strip()

        if not position:
            return jsonify({'success': False, 'error': '请输入目标岗位'}), 400
        if not resume:
            return jsonify({'success': False, 'error': '请输入简历内容'}), 400
        if len(resume) < 50:
            return jsonify({'success': False, 'error': '简历内容过短，请至少输入50个字符'}), 400

        result = analyze_resume_with_llm(position, resume)
        return jsonify({'success': True, 'data': result})

    except Exception as e:
        print(f"❌ 分析接口异常: {str(e)}")
        return jsonify({'success': False, 'error': f'服务器错误：{str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'api_configured': bool(DEEPSEEK_API_KEY),
        'base_url': DEEPSEEK_BASE_URL,
        'model': DEEPSEEK_MODEL
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
