from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import json
import time
import hashlib
import gc
import httpx
from collections import defaultdict
# 延迟导入：OpenAI 只在调用 API 时才 import，节省启动内存 ~30MB
# from openai import OpenAI  # 移到 _call_llm 函数内部

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
_rate_window  = defaultdict(list)   # IP -> [timestamp, ...]
_daily_count  = defaultdict(int)    # IP:date -> count
RATE_LIMIT_PER_MIN  = 5
RATE_LIMIT_PER_DAY  = 30
MAX_RESUME_LEN      = 5000
MAX_POSITION_LEN    = 100


def _get_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()


def _check_rate_limit(ip: str):
    now   = time.time()
    today = time.strftime('%Y-%m-%d')
    key_day = f"{ip}:{today}"

    # 定期清理过期数据，防止字典无限增长占用内存
    if int(now) % 60 == 0:
        for k in list(_rate_window.keys()):
            if not _rate_window[k] or now - _rate_window[k][-1] > 3600:
                del _rate_window[k]
        for k in list(_daily_count.keys()):
            if ':' in k and k.split(':')[1] != today:
                del _daily_count[k]

    if _daily_count[key_day] >= RATE_LIMIT_PER_DAY:
        return False, '今日请求次数已达上限（30次），请明天再试'

    _rate_window[ip] = [t for t in _rate_window[ip] if now - t < 60]
    if len(_rate_window[ip]) >= RATE_LIMIT_PER_MIN:
        return False, '请求过于频繁，请稍等片刻再试'

    _rate_window[ip].append(now)
    _daily_count[key_day] += 1
    return True, ''


def _parse_json(text: str) -> dict:
    """从模型输出中提取 JSON，兼容 markdown 代码块包裹。"""
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


def _call_llm(system: str, prompt: str, temperature: float = 0.8, max_tokens: int = 2000) -> str:
    """调用大模型。关键：httpx 客户端用完即关，避免连接堆积导致内存溢出。"""
    # 延迟导入：只在真正调用时才导入 OpenAI，节省启动内存
    from openai import OpenAI
    
    http_client = httpx.Client(timeout=110.0)
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            http_client=http_client
        )
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    finally:
        http_client.close()   # 释放连接和内存
        gc.collect()          # 主动回收，缓解 512MB 内存压力


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

    if not position:     return jsonify({'success': False, 'error': '请输入目标岗位'}), 400
    if not resume:       return jsonify({'success': False, 'error': '请输入简历内容'}), 400
    if len(resume) < 50: return jsonify({'success': False, 'error': '简历内容过短，请至少输入 50 个字符'}), 400
    if not DEEPSEEK_API_KEY: return jsonify({'success': False, 'error': 'API 未配置，请联系管理员'}), 500

    seed = int(hashlib.md5(resume.encode()).hexdigest()[:8], 16) % 10000

    prompt = f"""你是科大讯飞职场情报平台的资深职业顾问，拥有全球招聘市场视野。

【目标岗位】{position}
【简历】{resume}
【多样性种子】{seed}

请完成：
1. 推荐公司：在全球范围内（含创业公司、外企、独角兽、行业隐形冠军等，不要只推头部大厂）推荐1家最匹配的公司，给出名称和推荐理由（60字内）。必须基于简历细节，避免千篇一律。
2. 简历修改建议：结合目标公司偏好，给出具体优化建议（突出的技能项目、关键词优化、增删内容、量化成果），至少200字分段说明。
3. 竞争力雷达评分：对6个维度打0-100整数分，评分要有区分度：skill_match技能匹配、experience工作经验、education教育背景、project_quality项目质量、expression简历表达、market_fit市场适配。

严格返回JSON，不要其他文字：
{{"recommended_company":"公司名（理由）","modification_advice":"至少200字分段建议","radar":{{"skill_match":75,"experience":60,"education":80,"project_quality":70,"expression":65,"market_fit":72}}}}"""

    try:
        raw = _call_llm("你是科大讯飞职场情报平台的AI职业顾问，只返回JSON，不加任何解释。", prompt, 0.85, 2000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except json.JSONDecodeError:
        return jsonify({'success': True, 'data': {
            'recommended_company': '分析完成',
            'modification_advice': raw,
            'radar': {'skill_match':70,'experience':65,'education':75,'project_quality':68,'expression':72,'market_fit':70}
        }})
    except Exception as e:
        print(f"❌ analyze error: {e}")
        return jsonify({'success': False, 'error': f'系统繁忙，请稍后重试'}), 500


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
    if len(resume) < 50:
        return jsonify({'success': False, 'error': '简历内容过短，请至少输入 50 个字符'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API 未配置'}), 500

    tgt = f'（目标公司：{company}）' if company else ''
    prompt = f"""你是科大讯飞职场情报平台的资深面试官，正在面试应聘「{position}」岗位的候选人{tgt}。

候选人简历：{resume}

根据简历生成5道定制化面试题，覆盖技术深度、项目经验、行为能力，要有针对性，不出泛泛的通用题。每题提供：题目、考察重点（一句话）、参考答案（150字左右，结合简历背景）、难度（初级/中级/高级）。

严格返回JSON：
{{"questions":[{{"id":1,"question":"题目","focus":"考察重点","answer":"参考答案","level":"中级"}}]}}"""

    try:
        raw = _call_llm("你是科大讯飞职场情报平台的AI面试官，只返回JSON，不加任何解释。", prompt, 0.8, 3000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except Exception as e:
        print(f"❌ interview error: {e}")
        return jsonify({'success': False, 'error': '系统繁忙，请稍后重试'}), 500


# ══════════════════════════════════════════════════════════
#  接口 3：行业薪资情报
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

    if not position:
        return jsonify({'success': False, 'error': '请输入岗位名称'}), 400
    if not DEEPSEEK_API_KEY:
        return jsonify({'success': False, 'error': 'API 未配置'}), 500

    prompt = f"""你是科大讯飞职场情报平台的薪酬分析师，请针对岗位「{position}」（目标市场：{region or '全球'}）提供薪资情报。

请提供：
1. 薪资分级：初级/中级/高级三档，单位万元/年，给出具体数字范围。
2. 全球代表性公司薪资对比：选5家有代表性的公司（覆盖科技巨头、独角兽、外企、国内头部等不同类型），标注是否热招。
3. 市场洞察：150字左右，含市场热度、趋势判断、核心影响因素、薪资提升建议。

严格返回JSON，不要其他文字：
{{"levels":[{{"level":"初级 (0-3年)","salary":"15-25万"}},{{"level":"中级 (3-6年)","salary":"25-45万"}},{{"level":"高级 (6年+)","salary":"45-80万"}}],"companies":[{{"name":"公司名","salary":"30-50万","note":"特点","hot":true}}],"insight":"150字市场洞察"}}"""

    try:
        raw = _call_llm("你是科大讯飞职场情报平台的薪酬分析师，只返回JSON，不加任何解释。", prompt, 0.7, 2000)
        return jsonify({'success': True, 'data': _parse_json(raw)})
    except Exception as e:
        print(f"❌ salary error: {e}")
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
