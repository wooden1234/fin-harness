# fin-agent-platform

金融 Multi-Agent 智能客服（8 周作品集）。

## 启动

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：<http://127.0.0.1:8000/health> · API 文档：<http://127.0.0.1:8000/docs>

复制 `.env.example` 为 `.env` 后按需修改（本地可先不连库）。
