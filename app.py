from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# DeepSeek API配置
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def analyze_resume_with_llm(position: str, resume: str) -> dict:
    """使用DeepSeek大模型分析简历并推荐公司"""
    
    if not DEEPSEEK_API_KEY:
        return {
            "recommended_company": "API配置错误",
            "modification_advice": "请配置DEEPSEEK_API_KEY环境变量"
        }
    
    try:
        # 明确清除环境变量中可能存在的代理设置，避免 proxies 参数冲突
        import httpx
        http_client = httpx.Client()
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            http_client=http_client
        )
        
        prompt = f"""你是一位资深的招聘顾问和职业规划专家。请根据以下信息，为求职者提供精准的公司推荐和简历优化建议。

【目标岗位】
{position}

【求职者简历】
{resume}

请完成以下任务：

1. **推荐最合适的公司**：基于求职者的技能、经验和目标岗位，推荐1家最匹配的公司（可以是知名互联网公司、AI公司、传统科技企业等），并简要说明推荐理由（50字以内）。

2. **简历修改建议**：提供具体的、可操作的简历优化方向，包括：
   - 应该突出哪些技能和项目经验
   - 需要补充或弱化哪些内容
   - 如何调整表述以匹配目标公司和岗位的要求
   - 建议的关键词和亮点

请以JSON格式返回，严格按照以下格式：
{{
  "recommended_company": "公司名称（推荐理由）",
  "modification_advice": "详细的简历修改建议，分段说明，至少200字"
}}

注意：
- 推荐的公司要真实存在且与岗位高度相关
- 建议要具体、可执行，避免空泛的建议
- 直接返回JSON，不要有其他解释文字"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的招聘顾问，擅长分析求职者背景并提供精准的职业建议。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 尝试解析JSON
        import json
        # 移除可能的markdown代码块标记
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        return {
            "recommended_company": result.get("recommended_company", "未知公司"),
            "modification_advice": result.get("modification_advice", "暂无建议")
        }
        
    except Exception as e:
        print(f"❌ LLM调用失败: {str(e)}")
        return {
            "recommended_company": "分析失败",
            "modification_advice": f"系统错误：{str(e)}"
        }

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """分析接口"""
    try:
        data = request.get_json()
        position = data.get('position', '').strip()
        resume = data.get('resume', '').strip()
        
        # 参数验证
        if not position:
            return jsonify({
                'success': False,
                'error': '请输入目标岗位'
            }), 400
        
        if not resume:
            return jsonify({
                'success': False,
                'error': '请输入简历内容'
            }), 400
        
        if len(resume) < 50:
            return jsonify({
                'success': False,
                'error': '简历内容过短，请至少输入50个字符'
            }), 400
        
        # 调用大模型分析
        result = analyze_resume_with_llm(position, resume)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'服务器错误：{str(e)}'
        }), 500

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'api_configured': bool(DEEPSEEK_API_KEY)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
