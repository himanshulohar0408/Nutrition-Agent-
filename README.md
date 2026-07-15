# 🥗 NutriGenie — AI Nutrition Agent

> **AI-powered personal nutrition assistant** built with Python Flask + IBM Watsonx.ai (Granite models)  
> Features: Chat UI · Meal Planner · BMI Calculator · Food Analyzer · Family Profiles · Dark Mode · Mobile Responsive

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Chat** | Real-time conversational nutrition advice powered by IBM Granite |
| 🍱 **Meal Planner** | Personalized Indian/global meal plans with calorie targets |
| ⚖️ **BMI Calculator** | BMI + TDEE using Mifflin-St Jeor formula |
| 🔬 **Food Analyzer** | Detailed nutritional breakdown of any food item |
| 👨‍👩‍👧 **Family Profiles** | Age-grouped nutrition advice for every family member |
| 🌙 **Dark Mode** | One-click theme toggle, persisted in localStorage |
| 📱 **Mobile First** | Responsive sidebar + bottom navigation for mobile |
| 🎛️ **AGENT_INSTRUCTIONS** | Centralized config block in `app.py` for full customization |

---

## 🗂️ Project Structure

```
nutrition-agent/
├── app.py               ← Flask backend + AGENT_INSTRUCTIONS config
├── requirements.txt     ← Python dependencies
├── .env.example         ← Environment variables template
├── .env                 ← Your actual credentials (never commit!)
├── .gitignore
├── README.md
└── templates/
    └── index.html       ← Full single-page frontend (HTML + CSS + JS)
```

---

## ⚡ Quick Start

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd nutrition-agent
```

### 2. Create a Python virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up credentials
```bash
# Copy the template
cp .env.example .env
```
Edit `.env` and fill in your real credentials:
```
IBM_API_KEY=your_ibm_api_key_here
IBM_PROJECT_ID=your_project_id_here
IBM_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=any-random-secret-string
```

### 5. Get IBM Watsonx.ai credentials

| Item | Where to get it |
|---|---|
| **IBM_API_KEY** | https://cloud.ibm.com → Manage → Access (IAM) → API keys → Create |
| **IBM_PROJECT_ID** | https://dataplatform.cloud.ibm.com → Your project → Manage → General |
| **IBM_URL** | Choose by region: `us-south`, `eu-de`, `jp-tok` |

> **Note:** Your IBM Cloud account must have access to **IBM Watsonx.ai** and the **Granite** model. Free tier (Lite plan) includes limited tokens.

### 6. Run the application
```bash
python app.py
```

Open your browser at **http://localhost:5000** 🎉

---

## 🎛️ Customizing AGENT_INSTRUCTIONS

Open `app.py` and find the `AGENT_INSTRUCTIONS` dict near the top. This is your single control panel:

```python
AGENT_INSTRUCTIONS = {
    # Agent identity
    "name": "NutriGenie",
    "tagline": "Your Personal AI Nutrition Expert",
    "tone": "warm, encouraging, and scientifically grounded",

    # Diet specialization
    "diet_specialization": "general",   # "vegetarian", "keto", "diabetic", etc.

    # Indian food preferences
    "indian_food_enabled": True,
    "preferred_cuisines": ["North Indian", "South Indian", ...],

    # Safety rules (always enforced)
    "safety_rules": [...],

    # Response behavior
    "use_emojis_in_response": True,
    "include_hindi_food_names": True,
    "max_response_tokens": 900,

    # Model
    "model_id": "ibm/granite-3-3-8b-instruct",
    "temperature": 0.7,
    ...
}
```

**Common customizations:**

| Change | Field |
|---|---|
| Switch to a vegan agent | `"diet_specialization": "vegan"` |
| Disable Indian food mode | `"indian_food_enabled": False` |
| More concise responses | `"max_response_tokens": 400` |
| More creative responses | `"temperature": 0.9` |
| Different Granite model | `"model_id": "ibm/granite-3-8b-instruct"` |
| Add a safety rule | Append to `"safety_rules"` list |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main web application |
| `POST` | `/api/chat` | AI chat message |
| `POST` | `/api/bmi` | BMI + TDEE calculation |
| `POST` | `/api/meal-plan` | Generate meal plan |
| `POST` | `/api/analyze-food` | Food nutritional analysis |
| `POST` | `/api/family-advice` | Family nutrition guide |
| `POST` | `/api/clear-chat` | Clear chat session |
| `GET` | `/api/health` | Health check |

### Example: Chat API
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What should I eat for breakfast?", "context": {"age": 28, "goal": "Weight Loss"}}'
```

---

## 🚀 Deployment

### Option A — Render.com (Recommended, Free Tier)
1. Push your code to GitHub (make sure `.env` is in `.gitignore`)
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `gunicorn app:app`
6. Add Environment Variables from your `.env` file
7. Deploy 🚀

### Option B — Railway.app
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
```
Set env vars in Railway dashboard.

### Option C — IBM Code Engine
```bash
# Build Docker image
docker build -t nutrition-agent .
# Push to IBM Container Registry
ibmcloud cr push icr.io/<namespace>/nutrition-agent
# Deploy to Code Engine
ibmcloud ce application create --name nutrition-agent --image icr.io/<namespace>/nutrition-agent
```

### Option D — Local Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```
```bash
docker build -t nutrition-agent .
docker run -p 5000:5000 --env-file .env nutrition-agent
```

---

## 🔒 Security Notes

- Never commit `.env` to version control
- Use a strong random `FLASK_SECRET_KEY` in production
- Set `FLASK_DEBUG=false` in production
- Consider rate limiting for production deployments

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, Flask 3.x |
| **AI Model** | IBM Watsonx.ai — `ibm/granite-3-3-8b-instruct` |
| **Frontend** | HTML5, CSS3, Vanilla JS, Bootstrap 5.3 |
| **Icons** | Bootstrap Icons |
| **Deployment** | Gunicorn, Render / Railway / IBM Code Engine |

---

## 📝 License
MIT License — free to use, modify, and distribute.

---
Made with ❤️ + 🥗 using IBM Watsonx.ai Granite
