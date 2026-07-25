from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import time
import hashlib
import httpx
from collections import defaultdict
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ── API 配置 ──────────────────────────────────────────────
DEEPSEEK_API_KEY  = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = os.environ.get(
    'DEEPSEEK_API_HOST',
    'https://ws-ndrn7uqqjjn2ucra.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
)
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-pro')

# ── 安全：IP 频率限制 ─────────────────────────────────────
# 每个 IP 每分钟最多 5 次请求，每天最多 30 次
_rate_window  = defaultdict(list)   # IP -> [timestamp, ...]
_daily_count  = defaultdict(int)    # IP:date -> count
RATE_LIMIT_PER_MIN  = 5
RATE_LIMIT_PER_DAY  = 30
MAX_RESUME_LEN      = 5000          # 简历最大字符数，防止超长输入耗尽 token
MAX_POSITION_LEN    = 100

def _get_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()

def _check_rate_limit(ip: str) -> tuple[bool, str]:
    now   = time.time()
    today = time.strftime('%Y-%m-%d')
    key_day = f"{ip}:{today}"

    # 每天限制
    if _daily_count[key_day] >= RATE_LIMIT_PER_DAY:
        return False, '今日请求次数已达上限（30次），请明天再试'

    # 每分钟限制：保留最近 60s 的记录
    _rate_window[ip] = [t for t in _rate_window[ip] if now - t < 60]
    if len(_rate_window[ip]) >= RATE_LIMIT_PER_MIN:
        return False, '请求过于频繁，请稍等片刻再试'

    _rate_window[ip].append(now)
    _daily_count[key_day] += 1
    return True, ''


def _llm_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=httpx.Client()
    )


def _parse_json(text: str) -> dict:
    """从模型输出中提取 JSON，兼容 markdown 代码块包裹的情况。"""
    text = text.strip()
    if '```' in text:
        for part in text.split('```'):
            part = part.strip().lstrip('json').strip()
            if part.startswith('{'):
                text = part
                break
    return json.loads(text)


# ══════════════════════════════════════════════════════════
#  接口 1：简历分析 + 雷达图数据
# ══════════════════════════════════════════════════════════
@app.route('/api/analyze', methods=['POST'])
def analyze():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    data     = request.get_json() or {}
    position = data.get('position', '').strip()[:MAX_POSITION_LEN]
    resume   = data.get('resume',   '').strip()[:MAX_RESUME_LEN]

    if not position: return jsonify({'success': False, 'error': '请输入目标岗位'}), 400
    if not resume:   return jsonify({'success': False, 'error': '请输入简历内容'}), 400
    if len(resume) < 50: return jsonify({'success': False, 'error': '简历内容过短，请至少输入 50 个字符'}), 400
    if not DEEPSEEK_API_KEY: return jsonify({'success': False, 'error': 'API 未配置，请联系管理员'}), 500

    # 用简历内容的哈希值作为"种子"，引导模型给出差异化推荐
    resume_hash = int(hashlib.md5(resume.encode()).hexdigest()[:8], 16) % 10000

    prompt = f"""你是一位拥有全球视野的资深职业顾问，服务于科大讯飞职场情报平台。

【目标岗位】{position}
【求职者简历摘要】{resume}
【随机种子（影响推荐多样性）】{resume_hash}

任务说明：
1. 推荐公司：在全球范围内（包括创业公司、外企、国企、独角兽等，不要只推荐头部互联网大厂）推荐1家与该求职者背景最匹配的公司，给出公司名称和推荐理由（60字以内）。推荐必须基于简历的具体细节，不能千篇一律。

2. 简历修改建议：结合目标公司的招聘偏好，给出具体的简历优化建议，包括：
   - 需要突出的核心技能和项目经验
   - 表述方式和关键词优化
   - 需要补充或删减的内容
   - 量化成果的建议

3. 竞争力雷达图评分：对该简历在以下6个维度打分（0-100整数），评分要有区分度，不能全部集中在60-80区间：
   - skill_match（技能匹配度）
   - experience（工作经验）
   - education（教育背景）
   - project_quality（项目质量）
   - expression（简历表达）
   - market_fit（市场匹配度）

请严格返回以下JSON格式，不要有任何其他文字：
{{
  "recommended_company": "公司名称（推荐理由）",
  "modification_advice": "详细修改建议，至少200字，分段落说明",
  "radar": {{
    "skill_match": 75,
    "experience": 60,
    "education": 80,
    "project_quality": 70,
    "expression": 65,
    "market_fit": 72
  }}
}}"""

    try:
        client = _llm_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是科大讯飞职场情报平台的 AI 职业顾问，擅长全球招聘市场分析。只返回 JSON，不加任何解释。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=2000
        )
        result = _parse_json(resp.choices[0].message.content)
        return jsonify({'success': True, 'data': result})
    except json.JSONDecodeError:
        raw = resp.choices[0].message.content
        return jsonify({'success': True, 'data': {
            'recommended_company': '分析完成',
            'modification_advice': raw,
            'radar': {'skill_match':70,'experience':65,'education':75,'project_quality':68,'expression':72,'market_fit':70}
        }})
    except Exception as e:
        print(f"❌ analyze error: {e}")
        return jsonify({'success': False, 'error': f'系统错误：{str(e)}'}), 500


# ══════════════════════════════════════════════════════════
#  接口 2：AI 面试官模拟
# ══════════════════════════════════════════════════════════
@app.route('/api/interview', methods=['POST'])
def interview():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    data     = request.get_json() or {}
    position = data.get('position', '').strip()[:MAX_POSITION_LEN]
    resume   = data.get('resume',   '').strip()[:MAX_RESUME_LEN]
    company  = data.get('company',  '').strip()[:100]

    if not position or not resume:
        return jsonify({'success': False, 'error': '请填写岗位和简历'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API 未配置'}), 500

    prompt = f"""你是一位严格但专业的面试官，正在面试一名应聘「{position}」岗位的候选人{f'（目标公司：{company}）' if company else ''}。

候选人简历：
{resume}

请根据简历内容生成 5 道定制化面试题，题目要有针对性，覆盖技术深度、项目经验、行为能力三个维度，不要出泛泛的通用题目。

每道题都提供：
- 题目本身
- 考察重点（一句话）
- 高质量参考答案（150字左右，结合候选人简历背景）
- 难度等级：初级/中级/高级

请严格返回以下JSON格式：
{{
  "questions": [
    {{
      "id": 1,
      "question": "题目内容",
      "focus": "考察重点",
      "answer": "参考答案",
      "level": "中级"
    }}
  ]
}}"""

    try:
        client = _llm_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是科大讯飞职场情报平台的 AI 面试官，只返回 JSON，不加任何解释。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=3000
        )
        result = _parse_json(resp.choices[0].message.content)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        print(f"❌ interview error: {e}")
        return jsonify({'success': False, 'error': f'系统错误：{str(e)}'}), 500


# ══════════════════════════════════════════════════════════
#  接口 3：行业薪资情报
# ══════════════════════════════════════════════════════════
@app.route('/api/salary', methods=['POST'])
def salary():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    data       = request.get_json() or {}
    position   = data.get('position', '').strip()[:MAX_POSITION_LEN]
    experience = data.get('experience', '').strip()[:50]
    location   = data.get('location', '').strip()[:50]

    if not position:
        return jsonify({'success': False, 'error': '请输入岗位名称'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API 未配置'}), 500

    prompt = f"""你是一位掌握全球招聘市场数据的薪酬分析师，请针对以下岗位提供薪资情报报告。

岗位：{position}
工作年限：{experience or '不限'}
地区：{location or '全球'}

请提供：
1. 薪资分级（初级/中级/高级三档，单位：万元/年，列出具体数字范围）
2. 全球代表性公司薪资对比（选5家有代表性的公司，覆盖不同类型：科技巨头、独角兽、外企、国内头部等），标注是否热招
3. 市场洞察（一段话，包含市场热度、趋势判断、核心影响因素、薪资提升建议）

请严格返回以下JSON格式，不要有任何其他文字：
{{
  "levels": [
    {{"level": "初级 (0-3年)", "salary": "15-25万"}},
    {{"level": "中级 (3-6年)", "salary": "25-45万"}},
    {{"level": "高级 (6年+)", "salary": "45-80万"}}
  ],
  "companies": [
    {{"name": "公司名", "salary": "30-50万", "note": "特点说明", "hot": true}}
  ],
  "insight": "一段150字左右的市场洞察，包含市场热度、发展趋势、影响薪资的核心因素以及给求职者的薪资提升建议"
}}"""

    try:
        client = _llm_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是科大讯飞职场情报平台的薪酬分析师，只返回 JSON，不加任何解释。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        result = _parse_json(resp.choices[0].message.content)
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        print(f"❌ salary error: {e}")
        return jsonify({'success': False, 'error': f'系统错误：{str(e)}'}), 500


# ══════════════════════════════════════════════════════════
#  页面路由
# ══════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'api_configured': bool(DEEPSEEK_API_KEY),
        'model': DEEPSEEK_MODEL,
        'base_url': DEEPSEEK_BASE_URL
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
