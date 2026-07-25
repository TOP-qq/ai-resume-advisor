# AI 简历智能推荐系统 - 故障排查指南

## 错误：Internal Server Error

### 快速诊断步骤

#### 1. 查看 Render 日志
在 Render Dashboard → 你的服务 → **Logs** 标签页，查看最新的错误信息。

常见错误类型：
- `ModuleNotFoundError` → 依赖包缺失
- `No module named 'templates'` → 模板路径错误
- `DEEPSEEK_API_KEY not found` → 环境变量未配置
- `port already in use` → 端口冲突

#### 2. 检查环境变量
确保在 Render Dashboard → Environment 里配置了：
```
DEEPSEEK_API_KEY = sk-xxxxx（你的真实API Key）
```

#### 3. 检查文件结构
必须严格按照这个结构上传到 GitHub：
```
ai-resume-advisor/
├── app.py
├── requirements.txt
├── render.yaml
├── Procfile
├── .python-version
└── templates/
    └── index.html
```

**特别注意：templates/ 必须是文件夹，index.html 必须在里面**

---

## 如果你把日志发给我

请复制 Render Logs 里的最后 20-30 行错误信息，我能立即定位问题。

---

## 临时快速修复版本

如果你急需上线，我提供一个**超简化无依赖版本**，先让服务跑起来，回头再调试完整版。
