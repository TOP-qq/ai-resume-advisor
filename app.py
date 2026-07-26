from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import time
import hashlib
import gc
import httpx
from collections import defaultdict

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
_rate_window  = defaultdict(list)
_daily_count  = defaultdict(int)
RATE_LIMIT_PER_MIN  = 5
RATE_LIMIT_PER_DAY  = 50
MAX_RESUME_LEN      = 5000
MAX_POSITION_LEN    = 100


def _get_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()


def _check_rate_limit(ip: str):
    now   = time.time()
    today = time.strftime('%Y-%m-%d')
    key_day = f"{ip}:{today}"

    # 定期清理
    if int(now) % 60 == 0:
        for k in list(_rate_window.keys()):
            if not _rate_window[k] or now - _rate_window[k][-1] > 3600:
                del _rate_window[k]
        for k in list(_daily_count.keys()):
            if ':' in k and k.split(':')[1] != today:
                del _daily_count[k]

    if _daily_count[key_day] >= RATE_LIMIT_PER_DAY:
        return False, '今日请求次数已达上限，请明天再试'

    _rate_window[ip] = [t for t in _rate_window[ip] if now - t < 60]
    if len(_rate_window[ip]) >= RATE_LIMIT_PER_MIN:
        return False, '请求过于频繁，请稍候再试'

    _rate_window[ip].append(now)
    _daily_count[key_day] += 1
    return True, ''


def _parse_json(text: str) -> dict:
    """从模型输出中提取 JSON"""
    text = text.strip()
    if '```' in text:
        for part in text.split('```'):
            part = part.strip()
            if part.startswith('json'):
                part = part[4:].strip()
            if part.startswith('{'):
                text = part
                break
    return json.loads(text)


def _call_deepseek(system: str, user_prompt: str, temperature: float = 0.8, max_tokens: int = 2000) -> str:
    """直接用httpx调用DeepSeek API，不依赖openai库，节省28MB内存"""
    client = httpx.Client(timeout=120.0)
    try:
        response = client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    finally:
        client.close()
        gc.collect()


# ══════════════════════════════════════════════════════════
#  接口 1：简历分析 + 雷达图
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
    regions  = data.get('regions', [])       # 目标地区（多选）
    co_types = data.get('co_types', [])      # 公司类型（多选）
    intents  = data.get('intents', [])       # 求职意向（多选）

    if not position:     return jsonify({'success': False, 'error': '请输入目标岗位'}), 400
    if not resume:       return jsonify({'success': False, 'error': '请输入简历内容'}), 400
    if len(resume) < 50: return jsonify({'success': False, 'error': '简历内容至少50字'}), 400
    if not DEEPSEEK_API_KEY: return jsonify({'success': False, 'error': 'API未配置'}), 500

    seed = int(hashlib.md5(resume.encode()).hexdigest()[:8], 16) % 10000

    # 组装用户筛选偏好
    pref_parts = []
    if regions:  pref_parts.append(f"目标地区：{('、'.join(regions))[:100]}")
    if co_types: pref_parts.append(f"公司类型：{('、'.join(co_types))[:100]}")
    if intents:  pref_parts.append(f"求职意向：{('、'.join(intents))[:100]}")
    pref_text = '\n'.join(pref_parts) if pref_parts else '不限（AI自由推荐）'

    prompt = f"""你是科大讯飞AI职场大脑的资深顾问，拥有全球招聘市场视野。

【岗位】{position}
【简历】{resume}
【求职偏好】
{pref_text}
【多样性种子】{seed}

任务（必须严格结合上面的"求职偏好"来推荐，不能忽略用户的地区/公司类型/意向选择）：
1. 推荐公司：根据用户偏好推荐1家最匹配的公司（若用户限定了地区/类型，必须在该范围内推荐；未限定则全球范围含创业/外企/独角兽/隐形冠军）。给出公司名称+推荐理由60字内，理由要体现如何契合用户的偏好和简历细节。
2. 简历优化：结合目标公司，给出具体建议（突出技能、关键词优化、增删内容、量化成果），至少200字分段。
3. 竞争力雷达：6维0-100整数评分，要有区分度：skill_match技能匹配、experience工作经验、education教育背景、project_quality项目质量、expression简历表达、market_fit市场适配。
4. 行动清单：给出3条求职者当下可立即执行的具体行动建议，每条30字内。

严格返回JSON，不要其他文字：
{{"recommended_company":"公司名（理由）","modification_advice":"至少200字分段建议","radar":{{"skill_match":75,"experience":60,"education":80,"project_quality":70,"expression":65,"market_fit":72}},"action_items":["行动1","行动2","行动3"]}}"""

    try:
        raw = _call_deepseek("你是科大讯飞AI职场大脑的职业顾问，只返回JSON，不加解释。", prompt, 0.85, 2000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except json.JSONDecodeError:
        return jsonify({'success': True, 'data': {
            'recommended_company': '分析完成',
            'modification_advice': raw,
            'radar': {'skill_match':70,'experience':65,'education':75,'project_quality':68,'expression':72,'market_fit':70},
            'action_items': []
        }})
    except Exception as e:
        print(f"❌ analyze: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ══════════════════════════════════════════════════════════
#  接口 2：AI 面试官
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
    if len(resume) < 50:
        return jsonify({'success': False, 'error': '简历至少50字'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API未配置'}), 500

    tgt = f'（目标公司：{company}）' if company else ''
    prompt = f"""你是科大讯飞AI职场大脑的面试官，面试「{position}」岗位候选人{tgt}。

简历：{resume}

根据简历生成5道定制面试题，覆盖技术深度、项目经验、行为能力，要针对性，不出泛泛通用题。每题提供：题目、考察重点（一句话）、参考答案（150字左右结合简历背景）、难度（初级/中级/高级）、进阶追问（1个深入追问，20字内）。

严格返回JSON：
{{"questions":[{{"id":1,"question":"题目","focus":"考察重点","answer":"参考答案","level":"中级","follow_up":"追问内容"}}]}}"""

    try:
        raw = _call_deepseek("你是科大讯飞AI职场大脑的面试官，只返回JSON，不加解释。", prompt, 0.8, 3000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except Exception as e:
        print(f"❌ interview: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ══════════════════════════════════════════════════════════
#  接口 3：薪资情报
# ══════════════════════════════════════════════════════════
@app.route('/api/salary', methods=['POST'])
def salary():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    data     = request.get_json() or {}
    position = data.get('position', '').strip()[:MAX_POSITION_LEN]
    region   = data.get('region', '').strip()[:50]
    years    = data.get('years', [])       # 工作年限
    expectations = data.get('expectations', [])  # 薪资预期
    scales   = data.get('scales', [])      # 公司规模

    if not position:
        return jsonify({'success': False, 'error': '请输入岗位'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API未配置'}), 500

    # 组装筛选条件
    filters = []
    if years: filters.append(f"工作年限：{('、'.join(years))[:100]}")
    if expectations: filters.append(f"薪资预期：{('、'.join(expectations))[:100]}")
    if scales: filters.append(f"公司规模：{('、'.join(scales))[:100]}")
    filter_text = '\n'.join(filters) if filters else '不限'

    prompt = f"""你是科大讯飞AI职场大脑的薪酬分析师，针对「{position}」（市场：{region or '全球'}）提供薪资情报。

【筛选条件】
{filter_text}

提供（必须结合上述筛选条件给出精准数据，不能忽略用户的年限/预期/规模选择）：
1. 薪资分级：初级/中级/高级三档，万元/年，具体数字范围。若用户指定年限，优先按该年限分档。
2. 代表性公司薪资对比：5家（覆盖科技巨头、独角兽、外企、国内头部等，若用户指定规模，必须在该范围推荐），标注是否热招。
3. 市场洞察：150字，含市场热度、趋势判断、核心影响因素、薪资提升建议。
4. 谈薪话术：3条实用的薪资谈判话术/技巧，每条40字内，可直接用于和HR沟通。

严格返回JSON：
{{"levels":[{{"level":"初级(0-3年)","salary":"15-25万"}},{{"level":"中级(3-6年)","salary":"25-45万"}},{{"level":"高级(6年+)","salary":"45-80万"}}],"companies":[{{"name":"公司名","salary":"30-50万","note":"特点","hot":true}}],"insight":"150字市场洞察","negotiation_tips":["话术1","话术2","话术3"]}}"""

    try:
        raw = _call_deepseek("你是科大讯飞AI职场大脑的薪酬分析师，只返回JSON，不加解释。", prompt, 0.7, 2000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except Exception as e:
        print(f"❌ salary: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ══════════════════════════════════════════════════════════
#  接口 4：AI助手对话
# ══════════════════════════════════════════════════════════
@app.route('/api/assistant', methods=['POST'])
def assistant():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    data     = request.get_json() or {}
    question = data.get('question', '').strip()[:500]

    if not question:
        return jsonify({'success': False, 'error': '请输入问题'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API未配置'}), 500

    prompt = f"""用户问题：{question}

你是科大讯飞AI职场大脑的贴身助手，为职场人士提供快速、实用的建议。回答要：
- 简洁直接（150字以内）
- 给出2-3条可操作建议
- 语气专业但亲切

直接返回纯文本回答，不要JSON格式。"""

    try:
        answer = _call_deepseek("你是科大讯飞AI职场大脑的助手，简洁实用地回答职场问题。", prompt, 0.8, 500)
        return jsonify({'success': True, 'answer': answer.strip()})
    except Exception as e:
        print(f"❌ assistant: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ══════════════════════════════════════════════════════════
#  接口 5：职业规划路线图
# ══════════════════════════════════════════════════════════
@app.route('/api/roadmap', methods=['POST'])
def roadmap():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    data    = request.get_json() or {}
    current = data.get('current', '').strip()[:100]
    target  = data.get('target', '').strip()[:100]

    if not current or not target:
        return jsonify({'success': False, 'error': '请填写当前岗位和目标岗位'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API未配置'}), 500

    prompt = f"""你是科大讯飞AI职场大脑的职业规划师。用户当前岗位「{current}」，目标岗位「{target}」。

请规划一条清晰的职业进阶路线，分3-4个阶段。每个阶段包含：
- 阶段名称（如"夯实基础期"）
- 时间跨度（如"0-1年"）
- 核心目标（一句话）
- 需要掌握的关键技能（2-3个）
- 里程碑标志（达成什么算过关）

严格返回JSON：
{{"stages":[{{"name":"阶段名","duration":"时间跨度","goal":"核心目标","skills":["技能1","技能2"],"milestone":"里程碑"}}]}}"""

    try:
        raw = _call_deepseek("你是科大讯飞AI职场大脑的职业规划师，只返回JSON，不加解释。", prompt, 0.75, 2000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except Exception as e:
        print(f"❌ roadmap: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ══════════════════════════════════════════════════════════
#  接口 6：每日职场挑战题
# ══════════════════════════════════════════════════════════
@app.route('/api/challenge', methods=['POST'])
def challenge():
    ip = _get_ip()
    ok, msg = _check_rate_limit(ip)
    if not ok:
        return jsonify({'success': False, 'error': msg}), 429

    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API未配置'}), 500

    import random
    topics = ['面试技巧', '简历优化', '薪资谈判', '职场沟通', '时间管理', '团队协作', '职业规划', '情绪管理']
    topic = random.choice(topics)

    prompt = f"""你是科大讯飞AI职场大脑的出题官。请出1道关于「{topic}」的职场选择题，帮助职场人提升能力。

要求：题目有实际场景，4个选项，1个正确答案，附解析。

严格返回JSON：
{{"question":"题目","options":["A选项","B选项","C选项","D选项"],"answer":0,"explanation":"解析说明为什么这个答案正确"}}
（answer为正确选项的索引0-3）"""

    try:
        raw = _call_deepseek("你是科大讯飞AI职场大脑的出题官，只返回JSON，不加解释。", prompt, 0.9, 1000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except Exception as e:
        print(f"❌ challenge: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ── 页面路由 ──────────────────────────────────────────────
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
