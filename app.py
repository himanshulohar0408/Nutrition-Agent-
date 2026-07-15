"""
============================================================
  AI Nutrition Agent - Flask Backend
  Powered by IBM Watsonx.ai (Granite Models)
============================================================
"""

import os
import json
import re
import urllib.request
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ─────────────────────────────────────────────────────────────
#  AGENT INSTRUCTIONS  ← Customize everything here
# ─────────────────────────────────────────────────────────────
AGENT_INSTRUCTIONS = {

    # --- Identity & Tone ---
    "name": "NutriGenie",
    "tagline": "Your Personal AI Nutrition Expert",
    "tone": "warm, encouraging, and scientifically grounded",

    # --- Diet Specialization ---
    # Options: "general", "vegetarian", "vegan", "keto", "diabetic",
    #          "weight_loss", "muscle_gain", "heart_healthy", "ayurvedic"
    "diet_specialization": "general",

    # --- Indian Food Preferences ---
    "indian_food_enabled": True,
    "preferred_cuisines": [
        "North Indian", "South Indian", "Bengali", "Gujarati",
        "Maharashtrian", "Punjabi", "Kerala", "Street Food"
    ],
    "indian_staples": [
        "dal", "roti", "sabzi", "rice", "idli", "dosa", "poha",
        "upma", "khichdi", "rajma", "chole", "paneer", "curd",
        "lassi", "chai", "paratha", "biryani", "sambar", "rasam"
    ],

    # --- Safety Rules ---
    "safety_rules": [
        "Never recommend medications or medical treatments",
        "Always suggest consulting a registered dietitian for medical conditions",
        "Do not provide advice for serious conditions like eating disorders",
        "Flag extremely low-calorie diets (<1200 kcal for women, <1500 kcal for men) as unsafe",
        "Recommend medical consultation for diabetes, heart disease, kidney issues",
        "Never diagnose medical conditions",
    ],

    # --- Response Behavior ---
    "max_response_tokens": 900,
    "response_language": "English",
    "include_hindi_food_names": True,       # e.g., "dal (lentil soup)"
    "use_emojis_in_response": True,
    "provide_portion_sizes": True,
    "always_include_calories": True,

    # --- Default Meal Plan Settings ---
    "default_meals_per_day": 5,             # breakfast + 2 snacks + lunch + dinner
    "default_water_intake_liters": 2.5,
    "default_calorie_deficit_for_loss": 300,  # kcal below TDEE
    "default_calorie_surplus_for_gain": 300,  # kcal above TDEE

    # --- Family Profile Defaults ---
    "family_age_groups": {
        "child": (2, 12),
        "teen": (13, 17),
        "adult": (18, 59),
        "senior": (60, 120),
    },

    # --- Model Settings ---
    "model_id": "meta-llama/llama-3-3-70b-instruct",
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
# ─────────────────────────────────────────────────────────────


load_dotenv()

# ── USDA FoodData Central RAG ────────────────────────────
USDA_API_KEY = "DEMO_KEY"   # free demo key — no signup needed
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_DETAIL_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

# Nutrient IDs we care about
_NUTRIENT_IDS = {
    "Energy":        1008,
    "Protein":       1003,
    "Carbohydrates": 1005,
    "Total Fat":     1004,
    "Fiber":         1079,
    "Sugar":         2000,
    "Sodium":        1093,
    "Calcium":       1087,
    "Iron":          1089,
    "Vitamin C":     1162,
}

# Keywords that signal a food/nutrition query worth RAG-ing
_FOOD_QUERY_PATTERNS = re.compile(
    r"\b(calories|calorie|protein|carb|fat|fiber|nutrition|nutritional|"
    r"macro|vitamin|mineral|kcal|nutrient|how much|how many|"
    r"analyze|analyse|what is in|per 100g|per serving|"
    r"paneer|dal|rice|roti|idli|dosa|chicken|banana|almond|"
    r"egg|milk|curd|yogurt|spinach|broccoli|oats|wheat)\b",
    re.IGNORECASE,
)


def fetch_usda_nutrition(food_query: str) -> dict | None:
    """Fetch top food match from USDA FoodData Central and return key nutrients."""
    try:
        params = urllib.parse.urlencode({
            "query": food_query,
            "api_key": USDA_API_KEY,
            "pageSize": 1,
            "dataType": "Foundation,SR Legacy,Survey (FNDDS)",
        })
        req = urllib.request.Request(
            f"{USDA_SEARCH_URL}?{params}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())

        foods = data.get("foods", [])
        if not foods:
            return None

        food = foods[0]
        nutrients_raw = {n["nutrientId"]: n for n in food.get("foodNutrients", [])}

        nutrients = {}
        for name, nid in _NUTRIENT_IDS.items():
            entry = nutrients_raw.get(nid)
            if entry:
                nutrients[name] = f"{round(entry.get('value', 0), 1)} {entry.get('unitName', '')}"

        return {
            "source": "USDA FoodData Central",
            "food_name": food.get("description", food_query),
            "fdc_id": food.get("fdcId"),
            "data_type": food.get("dataType", ""),
            "nutrients_per_100g": nutrients,
        }
    except Exception:
        return None


def build_rag_context(food_query: str) -> str:
    """Return a formatted RAG block to inject into the prompt, or empty string."""
    data = fetch_usda_nutrition(food_query)
    if not data:
        return ""
    lines = [f"  {k}: {v}" for k, v in data["nutrients_per_100g"].items()]
    nutrient_block = "\n".join(lines)
    return (
        f"\n\n[VERIFIED NUTRITION DATA from {data['source']}]\n"
        f"Food: {data['food_name']} (per 100g)\n"
        f"{nutrient_block}\n"
        f"Use these exact figures in your answer. Do not contradict them."
    )


def extract_food_subject(message: str) -> str:
    """Best-effort extraction of the food item being asked about."""
    # Remove common question prefixes to isolate food name
    cleaned = re.sub(
        r"^(what (are|is)|how (much|many)|calories (in|of)|"
        r"nutrition(al)? (info|value|data|facts)? (of|for|in)|"
        r"analyze|analyse|tell me about)\s+",
        "", message, flags=re.IGNORECASE
    ).strip(" ?.,")
    # Cap at 50 chars to avoid sending full sentences to USDA
    return cleaned[:50] if cleaned else message[:50]

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nutrition-agent-secret-2024")

# ── IBM Watsonx.ai Setup ─────────────────────────────────────
IBM_API_KEY   = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
IBM_URL        = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")

_model = None   # lazy-initialized

def get_model() -> ModelInference:
    global _model
    if _model is None:
        credentials = Credentials(api_key=IBM_API_KEY, url=IBM_URL)
        params = {
            GenParams.MAX_NEW_TOKENS: AGENT_INSTRUCTIONS["max_response_tokens"],
            GenParams.TEMPERATURE:    AGENT_INSTRUCTIONS["temperature"],
            GenParams.TOP_P:          AGENT_INSTRUCTIONS["top_p"],
            GenParams.REPETITION_PENALTY: AGENT_INSTRUCTIONS["repetition_penalty"],
        }
        _model = ModelInference(
            model_id=AGENT_INSTRUCTIONS["model_id"],
            credentials=credentials,
            project_id=IBM_PROJECT_ID,
            params=params,
        )
    return _model


# ── Prompt Builder ────────────────────────────────────────────
def build_system_prompt(context: dict | None = None) -> str:
    ai = AGENT_INSTRUCTIONS
    safety = "\n".join(f"  - {r}" for r in ai["safety_rules"])
    cuisines = ", ".join(ai["preferred_cuisines"])
    staples  = ", ".join(ai["indian_staples"])
    ctx_str = ""
    if context:
        ctx_str = f"\nUser context: {json.dumps(context)}"

    return f"""You are {ai['name']}, {ai['tagline']}.
Tone: {ai['tone']}.
Diet specialization: {ai['diet_specialization']}.
{"Indian food is preferred. Include dishes from: " + cuisines + ". Common staples: " + staples if ai['indian_food_enabled'] else ""}
{"Include Hindi food names in parentheses where relevant." if ai['include_hindi_food_names'] else ""}
{"Use relevant food emojis to make responses friendly." if ai['use_emojis_in_response'] else ""}
{"Always provide portion sizes (grams/cups/pieces)." if ai['provide_portion_sizes'] else ""}
{"Always include calorie counts for food items." if ai['always_include_calories'] else ""}

Safety Rules (MUST follow):
{safety}

When asked for a meal plan, structure it as: Breakfast | Morning Snack | Lunch | Evening Snack | Dinner.
When giving calorie counts, be specific (e.g., "1 cup cooked rice ≈ 200 kcal").
Keep responses concise, practical, and motivating.{ctx_str}

Respond in {ai['response_language']} only."""


def build_prompt(user_message: str, history: list, context: dict | None = None,
                 rag_context: str = "") -> str:
    system = build_system_prompt(context)
    messages = [{"role": "system", "content": system}]
    for h in history[-6:]:   # last 6 exchanges for context window
        messages.append({"role": h["role"], "content": h["content"]})
    # Append RAG data to the user message when available
    user_content = user_message + rag_context if rag_context else user_message
    messages.append({"role": "user", "content": user_content})

    # Granite instruction format
    prompt = ""
    for m in messages:
        role = m["role"]
        if role == "system":
            prompt += f"<|system|>\n{m['content']}\n"
        elif role == "user":
            prompt += f"<|user|>\n{m['content']}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{m['content']}\n"
    prompt += "<|assistant|>\n"
    return prompt


# ── BMI & Calorie Helpers ────────────────────────────────────
def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    h_m = height_cm / 100
    bmi = round(weight_kg / (h_m ** 2), 1)
    if bmi < 18.5:
        cat, color = "Underweight", "#3b82d4"
    elif bmi < 25:
        cat, color = "Normal weight", "#22c55e"
    elif bmi < 30:
        cat, color = "Overweight", "#f59e0b"
    else:
        cat, color = "Obese", "#ef4444"
    return {"bmi": bmi, "category": cat, "color": color}


def calculate_tdee(weight_kg: float, height_cm: float, age: int,
                   gender: str, activity: str) -> dict:
    # Mifflin-St Jeor BMR
    if gender.lower() == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    multipliers = {
        "sedentary": 1.2,
        "light":     1.375,
        "moderate":  1.55,
        "active":    1.725,
        "very_active": 1.9,
    }
    mult = multipliers.get(activity, 1.55)
    tdee = round(bmr * mult)
    ai = AGENT_INSTRUCTIONS
    return {
        "bmr":  round(bmr),
        "tdee": tdee,
        "weight_loss": tdee - ai["default_calorie_deficit_for_loss"],
        "weight_gain":  tdee + ai["default_calorie_surplus_for_gain"],
        "maintenance":  tdee,
    }


def get_age_group(age: int) -> str:
    for group, (lo, hi) in AGENT_INSTRUCTIONS["family_age_groups"].items():
        if lo <= age <= hi:
            return group
    return "adult"


# ── Multi-Agent System ────────────────────────────────────────
AGENT_PROMPTS = {
    "nutrition": """You are the Nutrition Knowledge Agent of NutriGenie.
Your ONLY job: provide precise, verified nutritional facts.
- Always state values per 100g AND per common serving.
- Include macros (protein, carbs, fat, fiber) and key micros (vitamins, minerals).
- Reference USDA or ICMR data where possible.
- Do NOT give diet advice or meal plans — only facts.
- Use food emojis and include Hindi names in parentheses.
Respond in English only.""",

    "diet": """You are the Diet Recommendation Agent of NutriGenie.
Your ONLY job: create personalized, practical meal plans.
- Always structure plans as: Breakfast | Morning Snack | Lunch | Evening Snack | Dinner.
- Include portion sizes (grams/cups/pieces) and calorie counts for every item.
- Prefer Indian cuisine. Include Hindi food names.
- Tailor to the user's goal, dietary restrictions, and calorie target.
- Do NOT give medical advice — only meal recommendations.
Respond in English only.""",

    "health": """You are the Health Advisory Agent of NutriGenie.
Your ONLY job: preventive health and chronic disease diet guidance.
- Focus on evidence-based dietary strategies for: diabetes, heart disease, hypertension, kidney health, obesity.
- Always recommend consulting a doctor or registered dietitian for medical conditions.
- Suggest specific Indian foods that help manage each condition.
- Explain WHY each recommendation helps (glycemic index, sodium, saturated fat, etc.).
- Never recommend medications. Flag dangerous dietary practices.
Respond in English only.""",

    "foodlog": """You are the Food Log & Feedback Agent of NutriGenie.
Your ONLY job: analyze logged meals and give constructive daily feedback.
- Review the user's logged meals for the day.
- Calculate approximate total calories, protein, carbs, fat, fiber.
- Identify nutritional gaps or excesses.
- Suggest 1-2 specific improvements for tomorrow.
- Be encouraging and non-judgmental in tone.
- Use emojis and keep feedback concise (under 200 words).
Respond in English only.""",
}


def build_agent_prompt(agent_type: str, user_message: str,
                       context: dict | None = None, rag_context: str = "") -> str:
    system = AGENT_PROMPTS.get(agent_type, AGENT_PROMPTS["nutrition"])
    if context:
        system += f"\nUser context: {json.dumps(context)}"
    user_content = user_message + rag_context if rag_context else user_message
    return f"<|system|>\n{system}\n<|user|>\n{user_content}\n<|assistant|>\n"


def _run_agent(prompt: str) -> str:
    try:
        model = get_model()
        result = model.generate_text(prompt=prompt)
        return result.strip() if isinstance(result, str) else str(result)
    except Exception as e:
        return f"⚠️ Agent error: {str(e)[:100]}"


# ══════════════════════════════════════════════════════════════
#  TRUE AGENTIC AI ENGINE
#  Architecture:
#    User message
#      → Orchestrator LLM (decides which tools to call)
#      → Tool execution (USDA, food log, BMI, health DB)
#      → Specialist Agent LLM (uses tool results to answer)
#      → Agent-to-agent chain (nutrition facts → diet plan when needed)
#      → Final answer + full reasoning trace
# ══════════════════════════════════════════════════════════════

# ── Tool Registry ─────────────────────────────────────────────
TOOLS = {
    "search_usda": {
        "description": "Fetch verified nutritional data (calories, protein, carbs, fat, fiber, vitamins) for a food item from USDA FoodData Central.",
        "parameters": {"food": "string — name of the food item to look up"},
    },
    "get_food_log": {
        "description": "Retrieve the user's logged meals for today with running totals of calories and macros.",
        "parameters": {},
    },
    "calculate_bmi": {
        "description": "Calculate BMI and TDEE (daily calorie needs) given weight, height, age, gender, and activity level.",
        "parameters": {
            "weight": "number in kg",
            "height": "number in cm",
            "age": "integer",
            "gender": "male or female",
            "activity": "sedentary | light | moderate | active | very_active",
        },
    },
    "get_health_guidelines": {
        "description": "Get evidence-based dietary guidelines for a chronic condition (diabetes, heart, hypertension, kidney, obesity).",
        "parameters": {"condition": "string — one of: diabetes, heart, hypertension, kidney, obesity"},
    },
    "call_diet_agent": {
        "description": "Ask the Diet Recommendation Agent to create a personalized meal plan. Use this AFTER getting nutrition facts or health guidelines.",
        "parameters": {
            "goal": "string — e.g. weight loss, diabetes management",
            "calories": "integer — daily calorie target",
            "context": "string — any extra user context, dietary restrictions, prior agent output to incorporate",
        },
    },
}


def _tool_search_usda(food: str) -> dict:
    result = fetch_usda_nutrition(food)
    if result:
        return {"status": "ok", "data": result}
    return {"status": "not_found", "data": f"No USDA data found for '{food}'"}


def _tool_get_food_log(user_food_log: list) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    today_log = [e for e in user_food_log if e.get("date") == today]
    totals = {
        "calories": sum(e["calories"] for e in today_log),
        "protein":  round(sum(e["protein"]  for e in today_log), 1),
        "carbs":    round(sum(e["carbs"]    for e in today_log), 1),
        "fat":      round(sum(e["fat"]      for e in today_log), 1),
        "fiber":    round(sum(e["fiber"]    for e in today_log), 1),
    }
    meals = [f"{e['meal_type']}: {e['food']} ({e['quantity']}) — {e['calories']} kcal" for e in today_log]
    return {"status": "ok", "data": {"meals": meals, "totals": totals, "count": len(today_log)}}


def _tool_calculate_bmi(weight: float, height: float, age: int,
                        gender: str, activity: str) -> dict:
    bmi_data  = calculate_bmi(weight, height)
    tdee_data = calculate_tdee(weight, height, age, gender, activity)
    return {"status": "ok", "data": {**bmi_data, **tdee_data}}


def _tool_get_health_guidelines(condition: str) -> dict:
    cfg = CONDITION_CONFIGS.get(condition.lower())
    if not cfg:
        return {"status": "not_found", "data": f"Unknown condition: {condition}"}
    return {"status": "ok", "data": cfg}


def _tool_call_diet_agent(goal: str, calories: int, context: str) -> dict:
    prompt = build_agent_prompt(
        "diet",
        f"Create a personalized Indian meal plan.\nGoal: {goal}\nCalorie target: {calories} kcal/day\nContext: {context}",
    )
    reply = _run_agent(prompt)
    return {"status": "ok", "data": reply}


def execute_tool(tool_name: str, params: dict, food_log: list) -> dict:
    """Execute a tool by name with given params. Returns tool result dict."""
    try:
        if tool_name == "search_usda":
            return _tool_search_usda(params.get("food", ""))
        elif tool_name == "get_food_log":
            return _tool_get_food_log(food_log)
        elif tool_name == "calculate_bmi":
            return _tool_calculate_bmi(
                float(params.get("weight", 70)),
                float(params.get("height", 170)),
                int(params.get("age", 25)),
                params.get("gender", "male"),
                params.get("activity", "moderate"),
            )
        elif tool_name == "get_health_guidelines":
            return _tool_get_health_guidelines(params.get("condition", "diabetes"))
        elif tool_name == "call_diet_agent":
            return _tool_call_diet_agent(
                params.get("goal", "balanced nutrition"),
                int(params.get("calories", 2000)),
                params.get("context", ""),
            )
        else:
            return {"status": "error", "data": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"status": "error", "data": str(e)[:120]}


# ── Orchestrator ──────────────────────────────────────────────
_ORCHESTRATOR_SYSTEM = """You are the NutriGenie Orchestrator — the central controller of a multi-agent nutrition AI system.

Your job is to READ the user's message, DECIDE which tools to call, and PLAN the response strategy.

You have access to these tools:
{tool_list}

RULES:
1. Respond ONLY with a valid JSON object — no prose, no markdown, no explanation.
2. The JSON must have exactly these keys:
   - "intent": one-sentence description of what the user wants
   - "agent": which specialist agent should give the final answer — one of: nutrition | diet | health | foodlog | general
   - "tools": list of tool calls needed BEFORE the final agent responds. Each item: {{"tool": "tool_name", "params": {{...}}}}
   - "reasoning": 1-2 sentences explaining WHY you chose these tools and this agent
3. If no tools are needed, set "tools" to an empty list [].
4. For food nutrition questions → always call search_usda first.
5. For meal plan requests → call search_usda for key ingredients, then call_diet_agent.
6. For health condition questions → call get_health_guidelines first.
7. For food log / daily intake questions → call get_food_log first.
8. For BMI/calorie questions where weight+height are in the message → call calculate_bmi.

Example output:
{{"intent": "User wants to know protein content of paneer", "agent": "nutrition", "tools": [{{"tool": "search_usda", "params": {{"food": "paneer"}}}}], "reasoning": "User asked a nutrition fact question. USDA data will provide verified protein values."}}"""


def run_orchestrator(user_message: str, context: dict) -> dict:
    """Ask the orchestrator LLM to plan which tools and agent to use."""
    tool_list = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    )
    system = _ORCHESTRATOR_SYSTEM.format(tool_list=tool_list)
    prompt = f"<|system|>\n{system}\n<|user|>\n{user_message}\nUser context: {json.dumps(context)}\n<|assistant|>\n"

    raw = _run_agent(prompt)

    # Extract JSON from the response (model may wrap it in text)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback plan if orchestrator fails to produce valid JSON
    return {
        "intent": user_message[:80],
        "agent": "nutrition",
        "tools": [],
        "reasoning": "Orchestrator fallback — direct answer mode.",
    }


def run_agentic_pipeline(user_message: str, context: dict,
                         food_log: list, history: list) -> dict:
    """
    Full agentic loop:
      1. Orchestrator decides tools + agent
      2. Tools execute in order, collecting observations
      3. Specialist agent receives message + all tool observations
      4. Final answer returned with full trace
    """
    trace = []   # reasoning trace shown in UI

    # Step 1 — Orchestrate
    trace.append({"step": "orchestrator", "status": "thinking",
                  "detail": "Analyzing your request…"})
    plan = run_orchestrator(user_message, context)
    trace.append({
        "step": "orchestrator", "status": "done",
        "detail": f"Intent: {plan.get('intent', '?')}",
        "reasoning": plan.get("reasoning", ""),
        "agent_selected": plan.get("agent", "nutrition"),
    })

    # Step 2 — Execute tools
    tool_observations = []
    for tool_call in plan.get("tools", []):
        tool_name = tool_call.get("tool", "")
        params    = tool_call.get("params", {})
        trace.append({"step": "tool_call", "status": "running",
                      "detail": f"Calling tool: {tool_name}", "params": params})
        result = execute_tool(tool_name, params, food_log)
        tool_observations.append({
            "tool":   tool_name,
            "params": params,
            "result": result,
        })
        trace.append({
            "step":   "tool_result",
            "status": "done" if result["status"] == "ok" else "error",
            "detail": f"{tool_name} → {result['status']}",
            "data":   str(result["data"])[:200],
        })

    # Step 3 — Build enriched prompt for specialist agent
    agent_type = plan.get("agent", "nutrition")

    # Agent-to-agent communication: if diet agent was called as a tool,
    # pass its output to the final agent as enriched context
    observation_block = ""
    if tool_observations:
        obs_lines = []
        for obs in tool_observations:
            if obs["result"]["status"] == "ok":
                data = obs["result"]["data"]
                # If data is a dict (structured), pretty-print key fields
                if isinstance(data, dict):
                    if "nutrients_per_100g" in data:
                        nutrients = ", ".join(
                            f"{k}: {v}" for k, v in data["nutrients_per_100g"].items()
                        )
                        obs_lines.append(
                            f"[{obs['tool']}] {data.get('food_name','?')} per 100g: {nutrients}"
                        )
                    elif "meals" in data:
                        meals_str = "; ".join(data["meals"][:5])
                        obs_lines.append(
                            f"[{obs['tool']}] Today's log ({data['count']} entries): {meals_str}. "
                            f"Totals: {json.dumps(data['totals'])}"
                        )
                    elif "tdee" in data:
                        obs_lines.append(
                            f"[{obs['tool']}] BMI={data.get('bmi')}, TDEE={data.get('tdee')} kcal, "
                            f"Category={data.get('category')}"
                        )
                    elif "focus" in data:
                        obs_lines.append(
                            f"[{obs['tool']}] Health guidelines for {obs['params'].get('condition','?')}: "
                            f"Focus on {data['focus']}. Avoid: {data['avoid']}."
                        )
                    else:
                        obs_lines.append(f"[{obs['tool']}] {str(data)[:300]}")
                else:
                    obs_lines.append(f"[{obs['tool']}] {str(data)[:300]}")

        observation_block = (
            "\n\n[TOOL OBSERVATIONS — use these verified facts in your answer]\n"
            + "\n".join(obs_lines)
            + "\n[END TOOL OBSERVATIONS]"
        )

    # History context (last 4 exchanges)
    history_block = ""
    for h in history[-4:]:
        role = "User" if h["role"] == "user" else "Assistant"
        history_block += f"{role}: {h['content']}\n"

    enriched_message = user_message + observation_block
    if history_block:
        enriched_message = f"[Conversation history]\n{history_block}\n[Current message]\n{enriched_message}"

    agent_prompt = build_agent_prompt(agent_type, enriched_message, context)

    trace.append({"step": "specialist_agent", "status": "running",
                  "detail": f"Calling {agent_type} agent…"})

    final_reply = _run_agent(agent_prompt)

    trace.append({"step": "specialist_agent", "status": "done",
                  "detail": f"{agent_type} agent responded"})

    return {
        "reply":       final_reply,
        "agent_used":  agent_type,
        "plan":        plan,
        "tools_called": [t["tool"] for t in plan.get("tools", [])],
        "trace":       trace,
        "timestamp":   datetime.now().strftime("%I:%M %p"),
    }


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "family_profiles" not in session:
        session["family_profiles"] = []
    if "food_log" not in session:
        session["food_log"] = []
    return render_template("index.html", agent=AGENT_INSTRUCTIONS)


@app.route("/api/chat", methods=["POST"])
def chat():
    data     = request.get_json(force=True)
    message  = data.get("message", "").strip()
    context  = data.get("context", {})   # optional user profile

    if not message:
        return jsonify({"error": "Empty message"}), 400

    history = session.get("chat_history", [])

    # RAG: fetch real nutrition data if message looks like a food/nutrition query
    rag_context = ""
    rag_source  = None
    if _FOOD_QUERY_PATTERNS.search(message):
        food_subject = extract_food_subject(message)
        rag_data = fetch_usda_nutrition(food_subject)
        if rag_data:
            rag_context = build_rag_context(food_subject)
            rag_source  = rag_data["food_name"]

    prompt = build_prompt(message, history, context, rag_context)

    try:
        model  = get_model()
        result = model.generate_text(prompt=prompt)
        reply  = result.strip() if isinstance(result, str) else result
    except Exception as e:
        reply = (
            f"⚠️ I'm having trouble connecting to the AI service right now. "
            f"Please check your IBM API credentials. (Error: {str(e)[:80]})"
        )

    # persist to session history
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    session["chat_history"] = history[-20:]   # keep last 20 turns
    session.modified = True

    return jsonify({
        "reply":      reply,
        "timestamp":  datetime.now().strftime("%I:%M %p"),
        "rag_source": rag_source,   # non-null when real data was injected
    })


@app.route("/api/bmi", methods=["POST"])
def bmi_endpoint():
    data   = request.get_json(force=True)
    weight = float(data["weight"])
    height = float(data["height"])
    age    = int(data.get("age", 25))
    gender = data.get("gender", "male")
    activity = data.get("activity", "moderate")

    bmi_data  = calculate_bmi(weight, height)
    tdee_data = calculate_tdee(weight, height, age, gender, activity)

    return jsonify({**bmi_data, **tdee_data})


@app.route("/api/meal-plan", methods=["POST"])
def meal_plan():
    data    = request.get_json(force=True)
    profile = data.get("profile", {})
    goal    = data.get("goal", "balanced nutrition")
    calories = data.get("calories", 2000)

    prompt_text = (
        f"Generate a complete 1-day Indian meal plan for:\n"
        f"Goal: {goal}\n"
        f"Target calories: {calories} kcal\n"
        f"Profile: {json.dumps(profile)}\n\n"
        f"Format: 5 meals (Breakfast, Morning Snack, Lunch, Evening Snack, Dinner). "
        f"Include food name (Hindi name), portion size, and calories for each item. "
        f"End with daily totals: total calories, protein (g), carbs (g), fat (g), fiber (g)."
    )

    history = []
    full_prompt = build_prompt(prompt_text, history)
    try:
        model  = get_model()
        result = model.generate_text(prompt=full_prompt)
        plan   = result.strip() if isinstance(result, str) else result
    except Exception as e:
        plan = f"Error generating meal plan: {str(e)[:80]}"

    return jsonify({"plan": plan, "calories": calories, "goal": goal})


@app.route("/api/family-advice", methods=["POST"])
def family_advice():
    data    = request.get_json(force=True)
    members = data.get("members", [])

    if not members:
        return jsonify({"error": "No family members provided"}), 400

    member_summaries = []
    for m in members:
        age_group = get_age_group(int(m.get("age", 25)))
        member_summaries.append(
            f"- {m.get('name', 'Member')}, age {m.get('age')}, "
            f"{m.get('gender', 'unknown')}, {age_group}"
            f"{', ' + m.get('condition', '') if m.get('condition') else ''}"
        )

    members_text = "\n".join(member_summaries)
    prompt_text  = (
        f"Create a comprehensive family nutrition guide for:\n{members_text}\n\n"
        f"For each member, provide: recommended daily calories, key nutrients to focus on, "
        f"2-3 specific Indian food recommendations, and any special dietary notes. "
        f"End with 3 common family meals that work for everyone."
    )

    full_prompt = build_prompt(prompt_text, [])
    try:
        model  = get_model()
        result = model.generate_text(prompt=full_prompt)
        advice = result.strip() if isinstance(result, str) else result
    except Exception as e:
        advice = f"Error generating family advice: {str(e)[:80]}"

    return jsonify({"advice": advice, "member_count": len(members)})


@app.route("/api/analyze-food", methods=["POST"])
def analyze_food():
    data      = request.get_json(force=True)
    food_item = data.get("food", "").strip()

    if not food_item:
        return jsonify({"error": "No food item provided"}), 400

    prompt_text = (
        f"Provide a detailed nutritional analysis for: {food_item}\n"
        f"Include: calories per 100g and per common serving, "
        f"macros (protein, carbs, fat, fiber), key micronutrients, "
        f"health benefits, and best time to eat it. "
        f"Also suggest 2 healthy Indian-style recipes using this ingredient."
    )

    full_prompt = build_prompt(prompt_text, [])
    try:
        model  = get_model()
        result = model.generate_text(prompt=full_prompt)
        analysis = result.strip() if isinstance(result, str) else result
    except Exception as e:
        analysis = f"Error analyzing food: {str(e)[:80]}"

    return jsonify({"analysis": analysis, "food": food_item})


@app.route("/api/rag-food", methods=["POST"])
def rag_food():
    """Direct endpoint: fetch verified nutrition data from USDA for a food item."""
    data      = request.get_json(force=True)
    food_item = data.get("food", "").strip()

    if not food_item:
        return jsonify({"error": "No food item provided"}), 400

    result = fetch_usda_nutrition(food_item)
    if not result:
        return jsonify({"error": f"No USDA data found for '{food_item}'"}), 404

    return jsonify(result)


# ── Phase 1b: Preventive Health ──────────────────────────────
CONDITION_CONFIGS = {
    "diabetes": {
        "label": "Diabetes Management",
        "emoji": "🩸",
        "focus": "low glycemic index foods, fiber-rich meals, blood sugar control",
        "avoid": "white rice, maida, sugar, fruit juices, processed foods",
        "color": "#dc2626",
    },
    "heart": {
        "label": "Heart Health",
        "emoji": "❤️",
        "focus": "omega-3 rich foods, fiber, potassium, antioxidants, unsaturated fats",
        "avoid": "saturated fat, trans fat, excess sodium, red meat, fried foods",
        "color": "#ef4444",
    },
    "hypertension": {
        "label": "Hypertension (High BP)",
        "emoji": "🫀",
        "focus": "DASH diet, potassium, magnesium, low sodium, leafy greens",
        "avoid": "salt, pickles, papad, processed snacks, alcohol, caffeine",
        "color": "#7c3aed",
    },
    "kidney": {
        "label": "Kidney Health",
        "emoji": "🫘",
        "focus": "controlled protein, low potassium, low phosphorus, adequate hydration",
        "avoid": "high-protein foods, bananas, oranges, dairy excess, dark colas",
        "color": "#ea580c",
    },
    "obesity": {
        "label": "Weight Management",
        "emoji": "⚖️",
        "focus": "calorie deficit, high fiber, high protein, low glycemic, portion control",
        "avoid": "sugary drinks, fried foods, refined carbs, late-night eating",
        "color": "#2563eb",
    },
}


@app.route("/api/health-plan", methods=["POST"])
def health_plan():
    data      = request.get_json(force=True)
    condition = data.get("condition", "diabetes").lower()
    profile   = data.get("profile", {})
    days      = data.get("days", 1)

    cfg = CONDITION_CONFIGS.get(condition, CONDITION_CONFIGS["diabetes"])
    prompt_text = (
        f"You are a preventive health diet specialist.\n"
        f"Create a {days}-day meal plan for a patient managing: {cfg['label']}.\n"
        f"Patient profile: {json.dumps(profile)}\n\n"
        f"Dietary focus: {cfg['focus']}\n"
        f"Foods to avoid: {cfg['avoid']}\n\n"
        f"Requirements:\n"
        f"- Use Indian foods wherever possible\n"
        f"- 5 meals per day: Breakfast, Morning Snack, Lunch, Evening Snack, Dinner\n"
        f"- Include portion sizes, calories, and why each food helps this condition\n"
        f"- End with key dietary rules for {cfg['label']} management\n"
        f"- Always remind the user to consult their doctor"
    )
    reply = _run_agent(build_agent_prompt("health", prompt_text, profile))
    return jsonify({"plan": reply, "condition": condition, "config": cfg})


@app.route("/api/agent-chat", methods=["POST"])
def agent_chat():
    """Route a message to the most appropriate specialized agent."""
    data       = request.get_json(force=True)
    message    = data.get("message", "").strip()
    agent_type = data.get("agent", "nutrition")   # nutrition|diet|health|foodlog
    context    = data.get("context", {})

    if not message:
        return jsonify({"error": "Empty message"}), 400

    rag_context = ""
    rag_source  = None
    if _FOOD_QUERY_PATTERNS.search(message):
        food_subject = extract_food_subject(message)
        rag_data     = fetch_usda_nutrition(food_subject)
        if rag_data:
            rag_context = build_rag_context(food_subject)
            rag_source  = rag_data["food_name"]

    prompt = build_agent_prompt(agent_type, message, context, rag_context)
    reply  = _run_agent(prompt)

    return jsonify({
        "reply":      reply,
        "agent":      agent_type,
        "rag_source": rag_source,
        "timestamp":  datetime.now().strftime("%I:%M %p"),
    })


# ── True Agentic Chat Endpoint ───────────────────────────────
@app.route("/api/agentic-chat", methods=["POST"])
def agentic_chat():
    """
    Full agentic pipeline:
    Orchestrator → Tools → Specialist Agent → Answer + Trace
    """
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    context = data.get("context", {})

    if not message:
        return jsonify({"error": "Empty message"}), 400

    history  = session.get("chat_history", [])
    food_log = session.get("food_log", [])

    result = run_agentic_pipeline(message, context, food_log, history)

    # Persist to history
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": result["reply"]})
    session["chat_history"] = history[-20:]
    session.modified = True

    return jsonify(result)


# ── Phase 2: Food Log / Diet Tracking ────────────────────────
@app.route("/api/food-log/add", methods=["POST"])
def food_log_add():
    data      = request.get_json(force=True)
    food_text = data.get("food", "").strip()
    quantity  = data.get("quantity", "1 serving")
    meal_type = data.get("meal_type", "snack")   # breakfast|lunch|dinner|snack

    if not food_text:
        return jsonify({"error": "No food provided"}), 400

    # USDA lookup for real numbers
    usda = fetch_usda_nutrition(food_text)
    nutrients = usda["nutrients_per_100g"] if usda else {}

    def _num(key):
        val = nutrients.get(key, "0 ")
        try:
            return float(val.split()[0])
        except Exception:
            return 0.0

    entry = {
        "id":        datetime.now().strftime("%H%M%S%f"),
        "food":      food_text,
        "quantity":  quantity,
        "meal_type": meal_type,
        "time":      datetime.now().strftime("%I:%M %p"),
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "calories":  round(_num("Energy")),
        "protein":   round(_num("Protein"), 1),
        "carbs":     round(_num("Carbohydrates"), 1),
        "fat":       round(_num("Total Fat"), 1),
        "fiber":     round(_num("Fiber"), 1),
        "source":    usda["food_name"] if usda else food_text,
        "verified":  usda is not None,
    }

    log = session.get("food_log", [])
    log.append(entry)
    session["food_log"] = log[-50:]   # keep last 50 entries
    session.modified = True

    return jsonify({"entry": entry, "total_entries": len(session["food_log"])})


@app.route("/api/food-log/today", methods=["GET"])
def food_log_today():
    today = datetime.now().strftime("%Y-%m-%d")
    log   = session.get("food_log", [])
    today_log = [e for e in log if e.get("date") == today]

    totals = {
        "calories": sum(e["calories"] for e in today_log),
        "protein":  round(sum(e["protein"]  for e in today_log), 1),
        "carbs":    round(sum(e["carbs"]    for e in today_log), 1),
        "fat":      round(sum(e["fat"]      for e in today_log), 1),
        "fiber":    round(sum(e["fiber"]    for e in today_log), 1),
    }
    return jsonify({"entries": today_log, "totals": totals, "date": today})


@app.route("/api/food-log/delete", methods=["POST"])
def food_log_delete():
    entry_id = request.get_json(force=True).get("id")
    log = session.get("food_log", [])
    session["food_log"] = [e for e in log if e.get("id") != entry_id]
    session.modified = True
    return jsonify({"status": "deleted"})


@app.route("/api/food-log/feedback", methods=["POST"])
def food_log_feedback():
    """AI feedback on today's logged meals via the Food Log Agent."""
    today = datetime.now().strftime("%Y-%m-%d")
    log   = session.get("food_log", [])
    today_log = [e for e in log if e.get("date") == today]

    if not today_log:
        return jsonify({"feedback": "No meals logged today yet. Start logging to get feedback! 🍽️"})

    tdee_target = request.get_json(force=True).get("tdee", 2000)
    totals = {
        "calories": sum(e["calories"] for e in today_log),
        "protein":  round(sum(e["protein"]  for e in today_log), 1),
        "carbs":    round(sum(e["carbs"]    for e in today_log), 1),
        "fat":      round(sum(e["fat"]      for e in today_log), 1),
        "fiber":    round(sum(e["fiber"]    for e in today_log), 1),
    }
    meals_summary = "\n".join(
        f"- {e['meal_type'].title()}: {e['food']} ({e['quantity']}) "
        f"~ {e['calories']} kcal, {e['protein']}g protein"
        for e in today_log
    )
    message = (
        f"Today's logged meals:\n{meals_summary}\n\n"
        f"Daily totals: {json.dumps(totals)}\n"
        f"Target TDEE: {tdee_target} kcal\n\n"
        f"Please give a brief nutritional feedback and 2 improvement suggestions."
    )
    prompt   = build_agent_prompt("foodlog", message)
    feedback = _run_agent(prompt)
    return jsonify({"feedback": feedback, "totals": totals})


@app.route("/api/food-log/clear", methods=["POST"])
def food_log_clear():
    session["food_log"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


@app.route("/api/clear-chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


@app.route("/api/health")
def health_check():
    return jsonify({
        "status":  "ok",
        "agent":   AGENT_INSTRUCTIONS["name"],
        "model":   AGENT_INSTRUCTIONS["model_id"],
        "version": "2.0.0",
    })


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n🥗  {AGENT_INSTRUCTIONS['name']} starting on http://localhost:{port}")
    print(f"📡  Model : {AGENT_INSTRUCTIONS['model_id']}")
    print(f"🌐  IBM URL: {IBM_URL}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
