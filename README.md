# LINE Grad Admissions Bot (FastAPI)

A **minimal, runnable** LINE Messaging API bot to query graduate-admissions info, built with **FastAPI**.
- Webhook endpoint: `/callback`
- Health check: `/healthz`
- Flex Message card template included
- Uses a simple CSV (can be exported from Google Sheets) as the data source

## Quickstart

### 1) Prerequisites
- Python 3.10+
- A LINE **Messaging API** channel (get **Channel secret** and **Channel access token** from LINE Developers Console)
- (Optional) A publicly readable CSV URL exported from **Google Sheets**

### 2) Setup
```bash
git clone <this-project>
cd line-grad-admissions-bot-fastapi
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN
```

### 3) Run locally
```bash
uvicorn app.main:app --reload --port 8000
```

Expose your localhost with a tunnel (e.g., **Cloudflared** or **ngrok**) and set the HTTPS URL as your LINE **Webhook URL**:
```
https://<your-tunnel-domain>/callback
```

Then **Verify Webhook** in the LINE Developers console.

### 4) Data source (CSV via Google Sheets)
- Put your programs data in `data/sample_programs.csv`, or host a public CSV from Google Sheets and set `DATA_CSV_URL` in `.env`.
- CSV columns (see the sample file):
  - `school, department, program, degree, deadline, english_required, gpa_min, portfolio_required, contact_url, aliases, notes`

### Try it
Send messages like:
- `台大 資工 截止`
- `商管 11月 截止`
- `台大資工 英文門檻`
The bot will parse simple keywords and reply with a Flex card or a list.

### Deploy
You can deploy to **Railway / Render / Fly.io / Cloud Run / Heroku**. Make sure to:
- set env vars from `.env`
- expose `app` on port 8080 (or the platform's port)
- update the Webhook URL in LINE console

## Stack (Technologies Used)
- **FastAPI** – web framework
- **Uvicorn** – ASGI server
- **line-bot-sdk** – LINE Messaging API SDK for Python
- **python-dotenv** – environment variables
- **pydantic** – data validation
- **requests** – fetch CSV if hosted online
- **pandas** – CSV parsing and simple filtering

## License
MIT
