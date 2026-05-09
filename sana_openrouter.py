"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA — Smart Autonomous Natural Agent  (OpenRouter Edition)                 ║
║  Edge-based Eutrophication Detection & Response System                       ║
║  AI backend: OpenRouter free Gemma models (no local GPU required)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES (all pip-installable, no Ollama needed):
    pip install customtkinter requests

SETUP:
    1. Get a free API key from https://openrouter.ai/keys
    2. Either:
       a) Set environment variable:   export OPENROUTER_API_KEY=sk-or-v1-...
       b) Or paste it into the key dialog that appears on first launch.

FREE MODELS AVAILABLE (all $0/token):
    google/gemma-3-27b-it:free    — best quality, 128K ctx
    google/gemma-4-31b-it:free    — latest Gemma 4, vision+reasoning
    google/gemma-4-26b-a4b-it:free — MoE, near-31B quality at 4B cost
    google/gemma-3-4b-it:free     — fastest, lightest

ARCHITECTURE CHANGES vs the Ollama edition:
    • `call_ollama_agent()` → `call_openrouter_agent()`
      Uses `requests.post()` to https://openrouter.ai/api/v1/chat/completions
      with Bearer token auth. Fully OpenAI-compatible payload format.
    • API key stored in `SimulationEngine.api_key`, set via GUI dialog or env var.
    • Model selector updated to OpenRouter free Gemma IDs.
    • `OPENROUTER_AVAILABLE` flag replaces `OLLAMA_AVAILABLE`.
    • All simulation physics, GUI layout, and queue architecture unchanged.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import os
import queue
import random
import threading
import time
from datetime import datetime, timedelta

try:
    import spidev
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 2000000
    spi.mode = 0
    SPI_AVAILABLE = True
except Exception as e:
    print(f"[WARN] SPI/LoRa not available: {e}")
    SPI_AVAILABLE = False

def w(r,v):
    if SPI_AVAILABLE: spi.xfer2([r|0x80,v])
def r(r):
    if SPI_AVAILABLE: return spi.xfer2([r&0x7F,0])[1]
    return 0
def b(r,n):
    if SPI_AVAILABLE: return spi.xfer2([r&0x7F]+[0]*n)[1:]
    return [0]*n

def init_lora():
    if not SPI_AVAILABLE: return
    w(0x01,0x00); time.sleep(0.01)
    w(0x01,0x80); time.sleep(0.01)
    w(0x06,0x6C); w(0x07,0x80); w(0x08,0x00)
    w(0x09,0x8F); w(0x1D,0x72); w(0x1E,0x74)
    w(0x33,0x27); w(0x3B,0x1D)
    w(0x0E,0x00); w(0x0F,0x00)

# ── Third-party ───────────────────────────────────────────────────────────────

import customtkinter as ctk

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] `python-dotenv` not found.  pip install python-dotenv for .env support")

# requests is almost certainly already installed; we fail gracefully if not
try:
    import requests as req_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] `requests` library not found.  pip install requests")

# ═════════════════════════════════════════════════════════════════════════════
#  OPENROUTER CONFIG
# ═════════════════════════════════════════════════════════════════════════════

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERRER = "https://github.com/sana-hydro-monitor"   # optional leaderboard header
OPENROUTER_APP_NAME = "SANA-Dashboard"

# Free Gemma models available on OpenRouter (as of April 2025)
FREE_MODELS = [
    "google/gemma-3-27b-it:free",    # recommended — best quality free
    "google/gemma-4-31b-it:free",    # Gemma 4, reasoning-capable
    "google/gemma-4-26b-a4b-it:free",# MoE variant
    "google/gemma-3-4b-it:free",     # fastest / lowest latency
]

# ═════════════════════════════════════════════════════════════════════════════
#  THEME & COLOUR CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":      "#0A0E1A",
    "bg_panel":     "#0D1220",
    "bg_card":      "#111827",
    "bg_terminal":  "#080C18",
    "border":       "#1E2D45",
    "border_bright":"#2A4070",

    "green":        "#00FF94",
    "green_dim":    "#00C070",
    "yellow":       "#FFD700",
    "orange":       "#FF8C00",
    "red":          "#FF3355",
    "red_dim":      "#8B0000",

    "cyan":         "#00D4FF",
    "cyan_dim":     "#0088AA",
    "blue":         "#4488FF",
    "purple":       "#AA66FF",

    "text_bright":  "#E8F4FF",
    "text_normal":  "#8BA0C0",
    "text_dim":     "#3D5070",
}

SEVERITY_COLORS = {
    "LOW":      COLORS["green"],
    "MODERATE": COLORS["yellow"],
    "HIGH":     COLORS["orange"],
    "CRITICAL": COLORS["red"],
    "IDLE":     COLORS["text_dim"],
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTOR PHYSICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class NodeState:
    """
    Holds the living environmental state for the single LoRa worker node.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.bloom_level = 0.0
        self.exg = 0.0
        self.gli = 0.0
        self.health = 100.0
        self.risk = "LOW"
        self.trend = "STABLE"
        self.last_action = "IDLE"
        self.aeration_ticks = 0

    def activate_aeration(self, duration_ticks: int):
        self.aeration_ticks = duration_ticks

def compute_health(bloom: float) -> float:
    return max(0.0, 100.0 - (bloom * 100.0))

def risk_level(bloom: float) -> str:
    if bloom < 0.2: return "LOW"
    elif bloom < 0.5: return "MODERATE"
    elif bloom < 0.8: return "HIGH"
    return "CRITICAL"

def compute_trend(current: float, previous: float) -> str:
    if current > previous + 0.02: return "RISING"
    if current < previous - 0.02: return "FALLING"
    return "STABLE"

# ═════════════════════════════════════════════════════════════════════════════
#  LORA TELEMETRY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def build_lora_payload(node: NodeState, sim_time: datetime, raw_json: str = "") -> dict:
    return {
        "node_id":        node.node_id,
        "timestamp":      sim_time.strftime("%H:%M:%S"),
        "date":           sim_time.strftime("%Y-%m-%d"),
        "severity":       node.risk,
        "bloom_level":    round(node.bloom_level, 3),
        "health":         round(node.health, 1),
        "aeration_active":node.aeration_ticks > 0,
        "raw":            raw_json,
        "indices": {
            "EXG": {"value": round(node.exg, 3)},
            "GLI": {"value": round(node.gli, 3)},
            "BLOOM": {"value": round(node.bloom_level, 3)}
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
#  AGENTIC AI CONTROLLER  —  OpenRouter Edition
# ═════════════════════════════════════════════════════════════════════════════

SANA_SYSTEM_PROMPT = """You are SANA-BRAIN, the agentic AI controller of the SANA (Smart Autonomous Natural Agent) environmental monitoring network deployed on a freshwater lake.

You receive real-time telemetry from an autonomous surface node measuring eutrophication via EXG and GLI indices.

Your job is to:
1. Analyze the incoming telemetry data
2. Determine the severity and nature of the bloom threat
3. Issue an automated response command
4. Generate a brief public safety bulletin

You MUST respond with ONLY a single valid JSON object — no markdown, no explanations outside the JSON. The JSON must have exactly these fields:

{
  "reasoning": "<2-3 sentences of chain-of-thought analysis of the key indices>",
  "action": "<one of: IDLE | ACTIVATE_AERATOR | DEPLOY_CHEMICALS | CRITICAL_HUMAN_INTERVENTION>",
  "severity": "<one of: LOW | MODERATE | HIGH | CRITICAL>",
  "bulletin": "<1-2 sentences for the public status board, clear plain English>"
}

Action selection guidelines:
- IDLE: LOW severity, no intervention needed
- ACTIVATE_AERATOR: MODERATE or HIGH — deploy dissolved-oxygen aerators to disrupt stratification
- DEPLOY_CHEMICALS: HIGH with cyanotoxin risk — apply algaecide/flocculant
- CRITICAL_HUMAN_INTERVENTION: CRITICAL — bloom is toxic, immediate human response required

Respond ONLY with the JSON object."""


def call_openrouter_agent(payload: dict, api_key: str, model: str = FREE_MODELS[0]) -> dict:
    """
    Send telemetry payload to OpenRouter and parse the response.
    """
    def rule_based_fallback(payload: dict, tag: str = "[FALLBACK]") -> dict:
        severity = payload.get("severity", "LOW")
        node     = payload["node_id"]
        bloom    = payload["bloom_level"]

        if severity == "CRITICAL":
            action  = "CRITICAL_HUMAN_INTERVENTION"
            bulletin = f"⚠ CRITICAL ALERT: {node} reports toxic bloom (BLOOM={bloom}). Avoid all water contact. Emergency services notified. {tag}"
            reason = f"BLOOM level {bloom} exceeds critical thresholds. Immediate human intervention required."
        elif severity == "HIGH":
            action  = "DEPLOY_CHEMICALS"
            bulletin = f"HIGH ALERT: {node} — elevated algal activity detected (BLOOM={bloom}). Algaecide deployment authorized. {tag}"
            reason = f"BLOOM level {bloom} indicates active bloom. Chemical intervention warranted."
        elif severity == "MODERATE":
            action  = "ACTIVATE_AERATOR"
            bulletin = f"MODERATE: {node} — algal levels rising (BLOOM={bloom}). Aerators activated. {tag}"
            reason = f"BLOOM level {bloom} shows early-stage bloom development. Aeration initiated."
        else:
            action  = "IDLE"
            bulletin = f"NOMINAL: {node} — water quality safe (BLOOM={bloom}). {tag}"
            reason = f"BLOOM index at safe levels ({bloom}). Monitoring continues."

        return {"reasoning": reason, "action": action, "severity": severity, "bulletin": bulletin}

    if not REQUESTS_AVAILABLE:
        return rule_based_fallback(payload, "[NO REQUESTS LIB]")

    if not api_key or not api_key.strip().startswith("sk-"):
        return rule_based_fallback(payload, "[NO API KEY]")

    idx = payload["indices"]
    key_data = {
        "node":          payload["node_id"],
        "time":          payload["timestamp"],
        "severity":      payload["severity"],
        "bloom_level":   payload["bloom_level"],
        "health":        payload["health"],
        "exg":           idx["EXG"]["value"],
        "gli":           idx["GLI"]["value"]
    }

    user_prompt = f"TELEMETRY PACKET:\\n{json.dumps(key_data, indent=2)}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://sana.ai",
        "X-Title": "SANA Lake Dashboard",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SANA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    try:
        resp = req_lib.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        
        # Strip markdown json block if present
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        parsed = json.loads(raw_text)

        for key in ("reasoning", "action", "severity", "bulletin"):
            if key not in parsed:
                raise ValueError(f"Missing field: '{key}'")

        return parsed

    except Exception as e:
        return rule_based_fallback(payload, f"[API ERROR: {e}]")

# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Orchestrates the single-node environment, LoRa telemetry processing,
    and AI inference scheduling.
    """
    def __init__(self):
        self.node = NodeState("worker_1")
        self.sim_time    = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
        self.tick_count  = 0
        self.telemetry_queue = queue.Queue()
        self.ai_queue        = queue.Queue()
        self.event_queue     = queue.Queue()
        self._running    = False
        self._paused     = False
        self._ai_pending = False
        self.tick_interval = 6.0
        self.model_name  = FREE_MODELS[0]
        self.api_key     = os.environ.get("OPENROUTER_API_KEY", "")

    def start(self):
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False
    def set_speed(self, interval_seconds: float):
        self.tick_interval = max(1.0, float(interval_seconds))

    def _run_loop(self):
        self.event_queue.put("[RX] LORA RADIO LISTENING...")
        if SPI_AVAILABLE:
            init_lora()
            
        while self._running:
            if self._paused:
                time.sleep(0.2)
                continue

            if SPI_AVAILABLE:
                w(0x01,0x85)
                irq = r(0x12)

                if irq & 0x40:
                    length = r(0x13)
                    fifo = r(0x10)

                    w(0x0D,fifo)
                    data = b(0x00,length)
                    w(0x12,0xFF)

                    text = bytes(data).decode(errors="ignore")
                    print(f"\\n[LORA RAW PAYLOAD]\\n{text}\\n")

                    try:
                        obj = json.loads(text)
                        indices = obj.get("indices", {})
                        bloom = float(indices.get("BLOOM", 0))
                        exg = float(indices.get("EXG", 0))
                        gli = float(indices.get("GLI", 0))
                        node_id = obj.get("node", "worker_1")
                    except:
                        bloom, exg, gli, node_id = 0, 0, 0, "unknown"
                        obj = {}

                    self.node.node_id = node_id
                    self.node.trend = compute_trend(bloom, self.node.bloom_level)
                    self.node.bloom_level = bloom
                    self.node.exg = exg
                    self.node.gli = gli
                    self.node.health = compute_health(bloom)
                    self.node.risk = risk_level(bloom)
                    
                    self.sim_time += timedelta(minutes=15)
                    self.tick_count += 1

                    self.event_queue.put(
                        f"[{self.sim_time.strftime('%H:%M')}] ── LORA PACKET ──"
                        f" {'🔴' if bloom>0.7 else '🟢'}"
                    )
                    
                    payload = build_lora_payload(self.node, self.sim_time, text)

                    self.event_queue.put(
                        f"  ↗ LoRa RX  {node_id} → QUEEN  "
                        f"[bloom={self.node.bloom_level:.2f}  sev={self.node.risk}]"
                    )
                    self.telemetry_queue.put(payload)

                    if not self._ai_pending:
                        self._ai_pending = True
                        ai_t = threading.Thread(
                            target=self._run_ai_analysis,
                            args=(payload,),
                            daemon=True
                        )
                        ai_t.start()

                    init_lora()

            time.sleep(0.02)

    def _run_ai_analysis(self, payload: dict):
        node_id = payload["node_id"]
        node    = self.node

        self.event_queue.put(
            f"  🤖 SANA-BRAIN → OpenRouter [{self.model_name.split('/')[-1]}] …"
        )

        ai_result = call_openrouter_agent(
            payload,
            api_key=self.api_key,
            model=self.model_name,
        )

        action = ai_result.get("action", "IDLE")
        if action == "ACTIVATE_AERATOR":
            node.activate_aeration(duration_ticks=6)
            self.event_queue.put(
                f"  ⚡ AERATOR ACTIVATED  {payload['node_id']}  (6-tick)"
            )
        elif action in ("DEPLOY_CHEMICALS", "CRITICAL_HUMAN_INTERVENTION"):
            node.activate_aeration(duration_ticks=10)
            self.node.trend = "FALLING"
            self.event_queue.put(
                f"  ☣  {action}  {payload['node_id']}  [ALERT DISPATCHED]"
            )

        self.ai_queue.put({
            "node_id":   payload["node_id"],
            "timestamp": payload["timestamp"],
            "result":    ai_result,
            "payload":   payload,
        })
        self._ai_pending = False


# ═════════════════════════════════════════════════════════════════════════════
#  API KEY DIALOG  (shown on startup if no key found in environment)
# ═════════════════════════════════════════════════════════════════════════════

class APIKeyDialog(ctk.CTkToplevel):
    """
    Modal dialog asking for the OpenRouter API key.
    Dismissed by clicking Save, pressing Enter, or closing (uses fallback mode).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("SANA — OpenRouter API Key")
        self.geometry("520x280")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg_panel"])
        self.grab_set()   # modal
        self.api_key = ""

        ctk.CTkLabel(
            self,
            text="◈ SANA  OpenRouter Configuration",
            font=ctk.CTkFont(family="Courier New", size=14, weight="bold"),
            text_color=COLORS["cyan"],
        ).pack(pady=(22, 4))

        ctk.CTkLabel(
            self,
            text=(
                "Paste your OpenRouter API key below.\n"
                "Get a free key at  openrouter.ai/keys\n"
                "Free Gemma models are available at $0/token."
            ),
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_normal"],
            justify="center",
        ).pack(pady=(0, 12))

        self.key_entry = ctk.CTkEntry(
            self,
            placeholder_text="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            width=440, height=36,
            font=ctk.CTkFont(family="Courier New", size=11),
            fg_color=COLORS["bg_terminal"],
            text_color=COLORS["text_bright"],
            border_color=COLORS["border_bright"],
            show="•",
        )
        self.key_entry.pack(pady=4)
        self.key_entry.bind("<Return>", lambda e: self._save())

        # Reveal toggle
        self._reveal = False
        ctk.CTkButton(
            self,
            text="👁 Show / Hide Key",
            width=160, height=26,
            font=ctk.CTkFont(family="Courier New", size=9),
            fg_color="transparent",
            text_color=COLORS["text_dim"],
            hover_color=COLORS["bg_card"],
            command=self._toggle_reveal,
        ).pack(pady=2)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=14)

        ctk.CTkButton(
            btn_row,
            text="✓  SAVE & CONNECT",
            width=180, height=34,
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=COLORS["green_dim"],
            hover_color=COLORS["green"],
            text_color=COLORS["bg_dark"],
            command=self._save,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_row,
            text="Skip (fallback mode)",
            width=160, height=34,
            font=ctk.CTkFont(family="Courier New", size=10),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["border"],
            text_color=COLORS["text_dim"],
            command=self._skip,
        ).pack(side="left", padx=8)

    def _toggle_reveal(self):
        self._reveal = not self._reveal
        self.key_entry.configure(show="" if self._reveal else "•")

    def _save(self):
        self.api_key = self.key_entry.get().strip()
        self.grab_release()
        self.destroy()

    def _skip(self):
        self.api_key = ""
        self.grab_release()
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
#  GUI DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class SANADashboard(ctk.CTk):
    """
    Main application window — SANA Hydro-Informatics Command Centre.

    Layout:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  HEADER  [logo · subtitle · status · clock]                         │
    ├──────────────┬──────────────────────┬──────────────────────────────┤
    │  SECTOR MAP  │  LIVE TELEMETRY      │  AI AGENT TERMINAL           │
    │              │  STREAM              │  [reasoning · commands]      │
    ├──────────────┤                      │                              │
    │  PUBLIC      │                      │                              │
    │  BULLETIN    │                      │                              │
    ├──────────────┴──────────────────────┴──────────────────────────────┤
    │  CONTROL BAR  [Play | Pause | Speed | Model | API key btn | Tick]  │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, engine: SimulationEngine):
        super().__init__()
        self.engine = engine

        self.title("SANA · Smart Autonomous Natural Agent  |  OpenRouter Edition  v1.1")
        self.geometry("1440x860")
        self.minsize(1200, 720)
        self.configure(fg_color=COLORS["bg_dark"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self._running    = True
        self._sim_active = False

        self._build_header()
        self._build_main_grid()
        self._build_control_bar()
        self._poll_queues()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── If no API key in environment, show dialog after window appears ────
        if not self.engine.api_key:
            self.after(400, self._prompt_api_key)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYOUT
    # ─────────────────────────────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                              corner_radius=0, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="◈ SANA",
            font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
            text_color=COLORS["cyan"],
        ).pack(side="left", padx=18, pady=8)

        ctk.CTkLabel(
            header,
            text="SMART AUTONOMOUS NATURAL AGENT  ·  FOG-COMPUTING SWARM  ·  EUTROPHICATION MONITOR",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"],
        ).pack(side="left", padx=4)

        # OpenRouter badge
        ctk.CTkLabel(
            header,
            text="[ OpenRouter ]",
            font=ctk.CTkFont(family="Courier New", size=9),
            text_color=COLORS["purple"],
        ).pack(side="left", padx=8)

        self.clock_label = ctk.CTkLabel(
            header, text="",
            font=ctk.CTkFont(family="Courier New", size=12),
            text_color=COLORS["text_normal"],
        )
        self.clock_label.pack(side="right", padx=18)
        self._update_clock()

        self.status_dot = ctk.CTkLabel(
            header, text="● OFFLINE",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            text_color=COLORS["text_dim"],
        )
        self.status_dot.pack(side="right", padx=12)

        # API key status indicator (right side)
        self.key_status_label = ctk.CTkLabel(
            header, text="🔑 NO KEY",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["yellow"],
        )
        self.key_status_label.pack(side="right", padx=8)
        self._refresh_key_status()

    def _build_main_grid(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.columnconfigure(2, weight=3)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        left_col = ctk.CTkFrame(main, fg_color="transparent")
        left_col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,4))
        left_col.rowconfigure(0, weight=3)
        left_col.rowconfigure(1, weight=2)

        self._build_sector_map(left_col)
        self._build_bulletin_board(left_col)
        self._build_telemetry_panel(main)
        self._build_ai_terminal(main)

    def _build_sector_map(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=0, sticky="nsew", pady=(0,4))

        tb = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=0, height=26)
        tb.pack(fill="x", side="top")
        tb.pack_propagate(False)
        ctk.CTkLabel(tb, text="  ⬡ SECTOR MAP  ·  LAKE OVERVIEW",
                     font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                     text_color=COLORS["cyan_dim"], anchor="w").pack(side="left", fill="y")

        self.map_canvas = ctk.CTkCanvas(content, bg=COLORS["bg_dark"], highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.map_canvas.bind("<Configure>", self._redraw_map)

        self.node_positions = [(0.25,0.28),(0.72,0.28),(0.25,0.72),(0.72,0.72)]

    def _redraw_map(self, event=None):
        c = self.map_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10: return

        cx, cy = w * 0.5, h * 0.5
        rx, ry = w * 0.36, h * 0.42
        c.create_oval(cx-rx, cy-ry, cx+rx, cy+ry, fill="#0A1828", outline=COLORS["border_bright"], width=1)
        c.create_text(cx, cy-50, text="◈ LAKE", fill=COLORS["text_dim"], font=("Courier New", 10))

        node = self.engine.node
        severity = node.risk
        color = SEVERITY_COLORS.get(severity, COLORS["green"])
        pulse_r = 25 + int(node.bloom_level * 15)

        c.create_oval(cx-pulse_r, cy-pulse_r, cx+pulse_r, cy+pulse_r, fill="", outline=color, width=1, dash=(3,3))
        c.create_oval(cx-20, cy-20, cx+20, cy+20, fill=COLORS["bg_card"], outline=color, width=3)
        c.create_text(cx, cy, text="WORKER", fill=color, font=("Courier New", 10, "bold"))
        c.create_text(cx, cy+32, text=node.node_id, fill=COLORS["cyan"], font=("Courier New", 9, "bold"))
        c.create_text(cx, cy+45, text=severity, fill=color, font=("Courier New", 8, "bold"))

        bar_w, bar_h = 50, 6
        c.create_rectangle(cx-bar_w//2, cy+55, cx+bar_w//2, cy+55+bar_h, fill=COLORS["bg_terminal"], outline="")
        filled = int(bar_w * node.bloom_level)
        if filled > 0:
            c.create_rectangle(cx-bar_w//2, cy+55, cx-bar_w//2+filled, cy+55+bar_h, fill=color, outline="")

    def _poll_queues(self):
        processed = 0
        while not self.engine.telemetry_queue.empty() and processed < 8:
            payload = self.engine.telemetry_queue.get_nowait()
            self._render_telemetry(payload)
            processed += 1

        processed = 0
        while not self.engine.ai_queue.empty() and processed < 2:
            item = self.engine.ai_queue.get_nowait()
            self._render_ai_result(item)
            self._refresh_key_status()
            processed += 1

        processed = 0
        while not self.engine.event_queue.empty() and processed < 20:
            msg = self.engine.event_queue.get_nowait()
            self._append(self.ai_text, msg + "\n", "event")
            processed += 1

        if self._sim_active:
            self._redraw_map()
            self.tick_label.configure(
                text=f"TICK: {self.engine.tick_count:04d}  "
                     f"SIM: {self.engine.sim_time.strftime('%H:%M')}"
            )

        if self._running:
            self.after(100, self._poll_queues)

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, widget: ctk.CTkTextbox, text: str, tag: str = ""):
        widget.configure(state="normal")
        if tag:
            widget._textbox.insert("end", text, tag)
        else:
            widget._textbox.insert("end", text)
        widget._textbox.see("end")
        lc = int(widget._textbox.index("end-1c").split(".")[0])
        if lc > 600:
            widget._textbox.delete("1.0", f"{lc-500}.0")
        widget.configure(state="disabled")

    def _render_telemetry(self, payload: dict):
        node     = payload["node_id"]
        ts       = payload["timestamp"]
        sev      = payload["severity"]
        bloom    = payload["bloom_level"]
        sev_tag  = sev.lower()

        self._append(self.telem_text, f"\\n{'─'*52}\\n", "dim")
        self._append(self.telem_text, f"  ↗ {node}  @{ts}", "header")
        self._append(self.telem_text, f"  [{sev}]\\n", sev_tag)

        idx = payload["indices"]
        rows = [
            ("EXG",      f"{idx['EXG']['value']:.3f}"),
            ("GLI",      f"{idx['GLI']['value']:.3f}"),
            ("BLOOM",    f"{idx['BLOOM']['value']:.3f}"),
            ("HEALTH",   f"{payload['health']:.1f}%"),
            ("TREND",    payload["trend"]),
        ]

        for label, value in rows:
            if label == "BLOOM":
                v_tag = "critical" if idx['BLOOM']['value']>=0.8 else "high" if idx['BLOOM']['value']>=0.5 else "low"
            elif label == "HEALTH":
                v_tag = "critical" if payload['health']<=20 else "low"
            else:
                v_tag = "value"

            self._append(self.telem_text, f"    {label:<12}", "label")
            self._append(self.telem_text, f"{value}\\n", v_tag)
        
        self._append(self.telem_text, f"\\n    RAW: ", "dim")
        self._append(self.telem_text, f"{payload['raw']}\\n", "dim")

    def _render_ai_result(self, item: dict):
        node    = item["node_id"]
        ts      = item["timestamp"]
        result  = item["result"]
        action  = result.get("action", "IDLE")
        sev     = result.get("severity", "LOW")
        reason  = result.get("reasoning", "—")
        bulletin= result.get("bulletin", "—")
        sev_tag = sev.lower() if sev != "IDLE" else "dim"

        self._append(self.ai_text, f"\n{'═'*50}\n", "dim")
        self._append(self.ai_text, f"  SANA-BRAIN  {node}  @{ts}\n", "purple")
        self._append(self.ai_text, "  SEVERITY : ", "dim")
        self._append(self.ai_text, f"{sev}\n", sev_tag)
        self._append(self.ai_text, "  ACTION   : ", "dim")
        self._append(self.ai_text, f"{action}\n", "action")
        self._append(self.ai_text, "\n  REASONING:\n", "dim")
        self._append(self.ai_text, f"  {reason}\n", "reasoning")

        self._append(self.bulletin_text, f"\n[{ts}] ", "timestamp")
        self._append(self.bulletin_text, f"{node} ", "node")
        self._append(self.bulletin_text, f"[{sev}]", sev_tag)
        self._append(self.bulletin_text,
                     f"\n{bulletin}\n",
                     sev_tag if sev in ("CRITICAL","HIGH") else "")

    def _update_clock(self):
        self.clock_label.configure(
            text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._update_clock)

    def _on_close(self):
        self._running = False
        self.engine.stop()
        self.after(200, self.destroy)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SANA — Smart Autonomous Natural Agent  v1.1             ║")
    print("║  Backend : OpenRouter (free Gemma models)                ║")
    print("╠══════════════════════════════════════════════════════════╣")

    # Check environment for pre-set key
    env_key = os.environ.get("OPENROUTER_API_KEY", "")
    if env_key:
        print(f"║  API key  : found in environment ({env_key[:8]}…)              ║")
    else:
        print("║  API key  : not set — will prompt in GUI dialog          ║")

    if not REQUESTS_AVAILABLE:
        print("║  ⚠ WARNING: `requests` not installed — fallback mode     ║")
        print("║    Run:  pip install requests                             ║")

    print("╚══════════════════════════════════════════════════════════╝\n")

    engine = SimulationEngine()
    if env_key:
        engine.api_key = env_key

    app = SANADashboard(engine)
    app.mainloop()


if __name__ == "__main__":
    main()