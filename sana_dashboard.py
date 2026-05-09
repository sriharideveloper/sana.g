"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SANA — Smart Autonomous Natural Agent                                       ║
║  Edge-based Eutrophication Detection & Response System                       ║
║  Fog-Computing Swarm Simulation with Agentic AI Controller                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIES:
    pip install customtkinter ollama

USAGE:
    python sana_dashboard.py

    Make sure `ollama` is running locally with gemma4 or llama3:
        ollama serve
        ollama pull gemma4
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import queue
import random
import threading
import time
import cv2
from PIL import Image
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

# Try importing ollama; fall back gracefully if not installed
try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARN] `ollama` library not found. Running in fallback mode.")

# ── Model Configuration ──────────────────────────────────────────────────────
DEFAULT_MODEL = "gemma3:1b"   # Must match an installed model from `ollama list`

def validate_ollama_model(model_name: str) -> bool:
    """
    Verify that the requested model is actually installed in Ollama.
    Prevents silent fallback or 404 errors from wrong model names.
    Returns True if the model is found, False otherwise.
    """
    if not OLLAMA_AVAILABLE:
        return False
    try:
        models = ollama_lib.list()
        for m in models.models:
            installed_name = m.model
            # Clean up the name (e.g. 'gemma3:1b' or 'gemma3:latest' -> 'gemma3')
            clean_installed = installed_name.split(':')[0].lower()
            clean_target = model_name.split(':')[0].lower()
            
            if model_name == installed_name or clean_target == clean_installed:
                print(f"[OK] Model '{model_name}' found as '{installed_name}'")
                return True
        print(f"[ERROR] Model '{model_name}' NOT FOUND. Installed models: {[m.model for m in models.models]}")
        return False
    except Exception as e:
        print(f"[WARN] Could not validate model: {e}")
        return False

# ═════════════════════════════════════════════════════════════════════════════
#  THEME & COLOUR CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":      "#0A0E1A",   # Deep navy — main window background
    "bg_panel":     "#0D1220",   # Slightly lighter panel background
    "bg_card":      "#111827",   # Card / widget backgrounds
    "bg_terminal":  "#080C18",   # Ultra-dark terminal areas
    "border":       "#1E2D45",   # Subtle border lines
    "border_bright":"#2A4070",   # Highlighted borders

    "green":        "#00FF94",   # OK / Low severity
    "green_dim":    "#00C070",
    "yellow":       "#FFD700",   # Moderate severity
    "orange":       "#FF8C00",   # High severity
    "red":          "#FF3355",   # Critical severity
    "red_dim":      "#8B0000",

    "cyan":         "#00D4FF",   # Primary accent / headers
    "cyan_dim":     "#0088AA",
    "blue":         "#4488FF",   # Secondary accent
    "purple":       "#AA66FF",   # AI / LLM output accent

    "text_bright":  "#E8F4FF",   # Primary text
    "text_normal":  "#8BA0C0",   # Secondary text
    "text_dim":     "#3D5070",   # Disabled / background text

    "node_colors":  ["#00FF94", "#FFD700", "#FF8C00", "#FF3355"],
}

SEVERITY_COLORS = {
    "LOW":      COLORS["green"],
    "MODERATE": COLORS["yellow"],
    "HIGH":     COLORS["orange"],
    "CRITICAL": COLORS["red"],
    "IDLE":     COLORS["text_dim"],
}

# ═════════════════════════════════════════════════════════════════════════════
#  SECTOR PHYSICS ENGINE  (Simulated Environmental State)
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
    """
    Build the JSON payload containing purely OG LoRa data for the UI and AI.
    """
    payload = {
        "node_id":   node.node_id,
        "timestamp": sim_time.strftime("%H:%M:%S"),
        "date":      sim_time.strftime("%Y-%m-%d"),
        "severity":  node.risk,
        "bloom_level": round(node.bloom_level, 3),
        "health":    round(node.health, 1),
        "trend":     node.trend,
        "raw":       raw_json,
        "indices": {
            "EXG": {"value": round(node.exg, 3)},
            "GLI": {"value": round(node.gli, 3)},
            "BLOOM": {"value": round(node.bloom_level, 3)}
        }
    }
    return payload


# ═════════════════════════════════════════════════════════════════════════════
#  AGENTIC AI CONTROLLER  (Ollama Integration)
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


def call_ollama_agent(payload: dict, model: str = DEFAULT_MODEL) -> dict:
    """
    Send telemetry payload to the local Ollama LLM and parse the response.
    """
    def rule_based_fallback(payload: dict) -> dict:
        severity = payload.get("severity", "LOW")
        node     = payload["node_id"]
        bloom    = payload["bloom_level"]

        if severity == "CRITICAL":
            action  = "CRITICAL_HUMAN_INTERVENTION"
            bulletin = f"⚠ CRITICAL ALERT: {node} reports toxic bloom (BLOOM={bloom}). Avoid all water contact. Emergency services notified."
            reason = f"BLOOM level {bloom} exceeds critical thresholds. Immediate human intervention required."
        elif severity == "HIGH":
            action  = "DEPLOY_CHEMICALS"
            bulletin = f"HIGH ALERT: {node} — elevated algal activity detected (BLOOM={bloom}). Algaecide deployment authorized."
            reason = f"BLOOM level {bloom} indicates active bloom. Chemical intervention warranted."
        elif severity == "MODERATE":
            action  = "ACTIVATE_AERATOR"
            bulletin = f"MODERATE: {node} — algal levels rising (BLOOM={bloom}). Aerators activated."
            reason = f"BLOOM level {bloom} shows early-stage bloom development. Aeration initiated."
        else:
            action  = "IDLE"
            bulletin = f"NOMINAL: {node} — water quality safe (BLOOM={bloom})."
            reason = f"BLOOM index at safe levels ({bloom}). Monitoring continues."

        return {"reasoning": reason, "action": action, "severity": severity, "bulletin": bulletin}

    # ── Attempt real Ollama call ───────────────────────────────────────────────
    if not OLLAMA_AVAILABLE:
        return rule_based_fallback(payload)

    try:
        # Build a compact but information-rich prompt using available indices
        key_indices = {
            "node":     payload["node_id"],
            "time":     payload["timestamp"],
            "severity": payload["severity"],
            "bloom%":   payload["bloom_level"],
            "health%":  payload["health"],
            "EXG":      payload["indices"]["EXG"]["value"],
            "GLI":      payload["indices"]["GLI"]["value"],
            "trend":    payload["trend"],
        }

        user_message = (
            f"Telemetry received from {key_indices['node']} at {key_indices['time']}:\n"
            f"{json.dumps(key_indices, indent=2)}\n\n"
            f"Analyze this data and provide your JSON response."
        )

        response = ollama_lib.chat(
            model   = model,
            messages=[
                {"role": "system",  "content": SANA_SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            options={"temperature": 0.2, "num_predict": 512},
        )

        # ── Verify correct model was used ─────────────────────────────────────
        served_model = response["model"] if isinstance(response, dict) else getattr(response, "model", None)
        if served_model and model not in served_model:
            print(f"[WARN] Model mismatch! Requested '{model}' but got '{served_model}'")

        raw_text = response["message"]["content"].strip()

        # ── Log raw LLM response to terminal ──────────────────────────────────
        print(raw_text)

        # ── JSON extraction: strip markdown code fences if present ─────────────
        if "```" in raw_text:
            # Extract content between first ``` and last ```
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            raw_text = raw_text[start:end] if start != -1 else raw_text

        parsed = json.loads(raw_text)

        # Validate required fields; fall back on missing keys
        for key in ("reasoning", "action", "severity", "bulletin"):
            if key not in parsed:
                raise ValueError(f"Missing key '{key}' in LLM response")

        # Tag response with model metadata for traceability
        parsed["_model_used"] = served_model or model

        return parsed

    except json.JSONDecodeError as e:
        # LLM returned malformed JSON → use fallback but note the error
        fallback = rule_based_fallback(payload)
        fallback["reasoning"] = f"[JSON PARSE ERROR: {e}] " + fallback["reasoning"]
        return fallback

    except Exception as e:
        err_str = str(e)
        # ── Detect model-not-found specifically ───────────────────────────────
        if "404" in err_str or "not found" in err_str.lower():
            print(f"[CRITICAL] Model '{model}' not found on Ollama server! "
                  f"Run: ollama pull {model}")
            fallback = rule_based_fallback(payload)
            fallback["reasoning"] = (f"[MODEL NOT FOUND: '{model}'] "
                                     f"Run `ollama pull {model}` to fix. "
                                     + fallback["reasoning"])
            return fallback

        # Ollama server unreachable, timeout, etc.
        fallback = rule_based_fallback(payload)
        fallback["reasoning"] = f"[OLLAMA ERROR: {type(e).__name__}: {e}] " + fallback["reasoning"]
        return fallback


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE  (ties all subsystems together)
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
        self.model_name  = DEFAULT_MODEL

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
                # ── REAL LORA RADIO MODE (DITTO RX LOGIC) ──────────────────
                w(0x01,0x85)   # FORCE RX
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
            else:
                # ── SIMULATED LORA MODE (FOR WINDOWS/TESTING) ─────────────
                time.sleep(self.tick_interval)
                
                # Synthetic data generation logic
                if self.node.aeration_ticks > 0:
                    self.node.aeration_ticks -= 1
                    bloom_change = random.uniform(-0.06, -0.02)
                else:
                    bloom_change = random.uniform(-0.02, 0.05)

                bloom = max(0.0, min(1.0, self.node.bloom_level + bloom_change))
                exg = bloom * 0.8 + random.uniform(-0.02, 0.02)
                gli = (1.0 - bloom) * 0.6 + random.uniform(-0.02, 0.02)
                
                self.node.trend = compute_trend(bloom, self.node.bloom_level)
                self.node.bloom_level = bloom
                self.node.exg = exg
                self.node.gli = gli
                self.node.health = compute_health(bloom)
                self.node.risk = risk_level(bloom)
                
                self.sim_time += timedelta(minutes=15)
                self.tick_count += 1
                
                raw_sim = json.dumps({"node": "worker_1", "indices": {"BLOOM": round(bloom, 3), "EXG": round(exg, 3), "GLI": round(gli, 3)}})
                payload = build_lora_payload(self.node, self.sim_time, raw_sim)
                
                self.event_queue.put(f"[{self.sim_time.strftime('%H:%M')}] ── SIMULATED PACKET ──")
                self.telemetry_queue.put(payload)
                
                if not self._ai_pending:
                    self._ai_pending = True
                    ai_t = threading.Thread(target=self._run_ai_analysis, args=(payload,), daemon=True)
                    ai_t.start()

            time.sleep(0.02)

    def _run_ai_analysis(self, payload: dict):
        """
        Call the Ollama AI controller, parse the response, apply any
        interventions back to the sector state, and push results to ai_queue.
        """
        node_id = payload["node_id"]
        sector = self.node

        self.event_queue.put(
            f"  🤖 SANA-BRAIN analysing {payload['node_id']}..."
        )

        try:
            ai_result = call_ollama_agent(payload, model=self.model_name)

            # ── Apply AI action back to the physical sector state ─────────────────
            action = ai_result.get("action", "IDLE")
            if action == "ACTIVATE_AERATOR":
                sector.activate_aeration(duration_ticks=6)
                self.event_queue.put(
                    f"  ⚡ AERATOR ACTIVATED  {payload['node_id']}  "
                    f"(6-tick intervention)"
                )
            elif action in ("DEPLOY_CHEMICALS", "CRITICAL_HUMAN_INTERVENTION"):
                # Strong chemical intervention: larger bloom reduction
                sector.activate_aeration(duration_ticks=10)
                sector.trend = "FALLING"  # Correctly set string trend
                self.event_queue.put(
                    f"  ☣  {action}  {payload['node_id']}  "
                    f"[ALERT DISPATCHED]"
                )

            # Push result to GUI queue
            self.ai_queue.put({
                "node_id":   payload["node_id"],
                "timestamp": payload["timestamp"],
                "result":    ai_result,
                "payload":   payload,
            })
        except Exception as e:
            self.event_queue.put(f"  ⚠ AI THREAD ERROR: {e}")
            print(f"[ERROR] AI analysis failed: {e}")
        finally:
            self._ai_pending = False


# ═════════════════════════════════════════════════════════════════════════════
#  GUI DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

class SANADashboard(ctk.CTk):
    """
    Main application window — SANA Hydro-Informatics Command Centre.

    Layout (grid-based, dark-mode):
    ┌─────────────────────────────────────────────────────────────┐
    │  HEADER BAR                                                 │
    ├──────────────┬──────────────────────┬───────────────────────┤
    │  SECTOR MAP  │  LIVE TELEMETRY      │  AI AGENT TERMINAL    │
    │  (4 nodes)   │  STREAM              │                       │
    ├──────────────┴──────────────────────┤  (reasoning + cmds)   │
    │  PUBLIC BULLETIN BOARD              │                       │
    ├─────────────────────────────────────┴───────────────────────┤
    │  CONTROL BAR  [Play | Pause | Speed slider | Model select]  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, engine: SimulationEngine):
        super().__init__()
        self.engine = engine

        # ── Window config ─────────────────────────────────────────────────────
        self.title("SANA · Smart Autonomous Natural Agent  |  Eutrophication Monitor v1.0")
        self.geometry("1024x640")
        self.minsize(900, 500)
        self.configure(fg_color=COLORS["bg_dark"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # ── Internal state ────────────────────────────────────────────────────
        self._running         = True
        self._sim_active      = False
        self.bulletin_entries = []   # list of bulletin dicts for the board

        # ── Build all panels ──────────────────────────────────────────────────
        self._build_header()
        self._build_main_grid()
        self._build_control_bar()

        # ── Start GUI polling loop ────────────────────────────────────────────
        self._poll_queues()

        # ── Window close handler ─────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYOUT CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _build_header(self):
        """Top banner with logo, status indicator, and clock."""
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                              corner_radius=0, height=52)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Logo text
        logo = ctk.CTkLabel(
            header,
            text="◈ SANA",
            font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
            text_color=COLORS["cyan"],
        )
        logo.pack(side="left", padx=18, pady=8)

        sub = ctk.CTkLabel(
            header,
            text="SMART AUTONOMOUS NATURAL AGENT  ·  FOG-COMPUTING SWARM  ·  EUTROPHICATION MONITOR",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"],
        )
        sub.pack(side="left", padx=4)

        # Live clock (updates every second)
        self.clock_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(family="Courier New", size=12),
            text_color=COLORS["text_normal"],
        )
        self.clock_label.pack(side="right", padx=18)
        self._update_clock()

        # System status
        self.status_dot = ctk.CTkLabel(
            header,
            text="● OFFLINE",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            text_color=COLORS["text_dim"],
        )
        self.status_dot.pack(side="right", padx=12)

    def _build_main_grid(self):
        """Create the three-column main content area."""
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        main.columnconfigure(0, weight=2)   # Sector map + bulletins
        main.columnconfigure(1, weight=3)   # Telemetry stream
        main.columnconfigure(2, weight=3)   # AI terminal
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        # ── LEFT column ───────────────────────────────────────────────────────
        left_col = ctk.CTkFrame(main, fg_color="transparent")
        left_col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,4))
        left_col.rowconfigure(0, weight=3)
        left_col.rowconfigure(1, weight=2)
        left_col.rowconfigure(2, weight=2)  # Added row for Video

        self._build_sector_map(left_col)
        self._build_bulletin_board(left_col)
        self._build_camera_panel(left_col)

        # ── CENTRE column ─────────────────────────────────────────────────────
        self._build_telemetry_panel(main)

        # ── RIGHT column ──────────────────────────────────────────────────────
        self._build_ai_terminal(main)

    def _make_panel(self, parent, title: str, row: int, col: int,
                    rowspan=1, sticky="nsew", padx=(0,4), pady=(0,4)) -> ctk.CTkFrame:
        """Helper: create a labelled panel card."""
        outer = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                             corner_radius=6, border_width=1,
                             border_color=COLORS["border"])
        outer.grid(row=row, column=col, rowspan=rowspan,
                   sticky=sticky, padx=padx, pady=pady)

        title_bar = ctk.CTkFrame(outer, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        ctk.CTkLabel(
            title_bar, text=f"  {title}",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        content = ctk.CTkFrame(outer, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=6, pady=4)

        return content

    # ── SECTOR MAP ────────────────────────────────────────────────────────────

    def _build_sector_map(self, parent):
        """
        Visual lake map: a canvas with 4 node indicators positioned around
        a stylised water body.  Colors update in real-time based on severity.
        """
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=0, sticky="nsew", padx=(0,0), pady=(0,4))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  ⬡ SECTOR MAP  ·  LAKE OVERVIEW",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.map_canvas = ctk.CTkCanvas(
            content, bg=COLORS["bg_dark"],
            highlightthickness=0,
        )
        self.map_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.map_canvas.bind("<Configure>", self._redraw_map)

        # Node positions as fractions of canvas size (NW, NE, SW, SE)
        self.node_positions = [
            (0.25, 0.28),  # Node 1 — NW
            (0.72, 0.28),  # Node 2 — NE
            (0.25, 0.72),  # Node 3 — SW
            (0.72, 0.72),  # Node 4 — SE
        ]

        # Per-node label refs for dynamic updates
        self.map_node_items = {}

    def _redraw_map(self, event=None):
        """Redraw the entire map canvas (called on resize or state change)."""
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

    # ── BULLETIN BOARD ───────────────────────────────────────────────────────

    def _build_bulletin_board(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=1, column=0, sticky="nsew", padx=(0,0), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  📢 PUBLIC BULLETIN BOARD",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.bulletin_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="word", state="disabled",
        )
        self.bulletin_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_bulletin_tags()

    def _configure_bulletin_tags(self):
        """Set up colour tags on the bulletin textbox."""
        tb = self.bulletin_text._textbox
        for sev, color in SEVERITY_COLORS.items():
            tb.tag_configure(sev, foreground=color)
        tb.tag_configure("timestamp", foreground=COLORS["text_dim"])
        tb.tag_configure("node",      foreground=COLORS["cyan"])

    # ── TELEMETRY PANEL ──────────────────────────────────────────────────────

    def _build_telemetry_panel(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=1, rowspan=2, sticky="nsew",
                     padx=(0,4), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  📡 LIVE TELEMETRY STREAM  ·  LoRa RX → QUEEN NODE",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.telem_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="none", state="disabled",
        )
        self.telem_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_telem_tags()

        # Initial placeholder
        self._telem_append(
            "  SANA QUEEN NODE  —  WAITING FOR TELEMETRY\n"
            "  ─────────────────────────────────────────\n"
            "  Start simulation to begin receiving LoRa packets.\n\n",
            "dim"
        )

    def _configure_telem_tags(self):
        tb = self.telem_text._textbox
        tb.tag_configure("header",   foreground=COLORS["cyan"])
        tb.tag_configure("critical", foreground=COLORS["red"])
        tb.tag_configure("high",     foreground=COLORS["orange"])
        tb.tag_configure("moderate", foreground=COLORS["yellow"])
        tb.tag_configure("low",      foreground=COLORS["green"])
        tb.tag_configure("dim",      foreground=COLORS["text_dim"])
        tb.tag_configure("label",    foreground=COLORS["text_normal"])
        tb.tag_configure("value",    foreground=COLORS["text_bright"])

    # ── AI AGENT TERMINAL ────────────────────────────────────────────────────

    def _build_ai_terminal(self, parent):
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=0, column=2, rowspan=2, sticky="nsew",
                     padx=(0,0), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  🤖 SANA-BRAIN  ·  AGENTIC AI CONTROLLER",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["purple"], anchor="w",
        ).pack(side="left", fill="y")

        self.ai_text = ctk.CTkTextbox(
            content, fg_color=COLORS["bg_terminal"],
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_bright"],
            corner_radius=0, wrap="word", state="disabled",
        )
        self.ai_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._configure_ai_tags()

        self._ai_append(
            "  SANA-BRAIN OFFLINE\n"
            "  ──────────────────────────────────────────────\n"
            "  Awaiting first telemetry packet.\n"
            f"  Ollama status: {'AVAILABLE' if OLLAMA_AVAILABLE else '⚠ NOT FOUND (fallback mode)'}\n\n",
            "dim"
        )

    def _configure_ai_tags(self):
        tb = self.ai_text._textbox
        tb.tag_configure("header",   foreground=COLORS["purple"])
        tb.tag_configure("reasoning",foreground=COLORS["text_normal"])
        tb.tag_configure("action",   foreground=COLORS["cyan"])
        tb.tag_configure("critical", foreground=COLORS["red"])
        tb.tag_configure("high",     foreground=COLORS["orange"])
        tb.tag_configure("moderate", foreground=COLORS["yellow"])
        tb.tag_configure("low",      foreground=COLORS["green"])
        tb.tag_configure("dim",      foreground=COLORS["text_dim"])
        tb.tag_configure("event",    foreground=COLORS["blue"])

    # ── CONTROL BAR ──────────────────────────────────────────────────────────

    def _build_control_bar(self):
        """Bottom bar: Play/Pause, speed slider, model selector."""
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"],
                           corner_radius=0, height=50)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        # Left: play/pause
        self.play_btn = ctk.CTkButton(
            bar, text="▶  START",
            width=120, height=34,
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=COLORS["green_dim"],
            hover_color=COLORS["green"],
            text_color=COLORS["bg_dark"],
            command=self._toggle_simulation,
        )
        self.play_btn.pack(side="left", padx=12, pady=8)

        # Speed label + slider
        ctk.CTkLabel(
            bar, text="SPEED:",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_normal"],
        ).pack(side="left", padx=(12,2))

        self.speed_label = ctk.CTkLabel(
            bar, text="4s/tick",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["cyan"],
            width=60,
        )
        self.speed_label.pack(side="left", padx=(0,4))

        self.speed_slider = ctk.CTkSlider(
            bar, from_=1, to=20,
            width=180, height=18,
            command=self._on_speed_change,
            button_color=COLORS["cyan"],
            button_hover_color=COLORS["cyan_dim"],
            progress_color=COLORS["border_bright"],
            fg_color=COLORS["border"],
        )
        self.speed_slider.set(4)
        self.speed_slider.pack(side="left", padx=4)

        # Separator
        ctk.CTkLabel(
            bar, text="│",
            text_color=COLORS["border_bright"],
        ).pack(side="left", padx=12)

        # Model selector
        ctk.CTkLabel(
            bar, text="MODEL:",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_normal"],
        ).pack(side="left", padx=(0,4))

        self.model_var = ctk.StringVar(value="gemma3:1b")
        self.model_menu = ctk.CTkOptionMenu(
            bar,
            values=["gemma3:1b", "gemma4:1b", "gemma:4b", "llama3", "mistral", "phi3", "llama3.2"],
            variable=self.model_var,
            width=130, height=30,
            font=ctk.CTkFont(family="Courier New", size=10),
            fg_color=COLORS["bg_card"],
            button_color=COLORS["border_bright"],
            button_hover_color=COLORS["border"],
            text_color=COLORS["text_bright"],
            command=self._on_model_change,
        )
        self.model_menu.pack(side="left", padx=4)

        # Right: tick counter
        self.tick_label = ctk.CTkLabel(
            bar,
            text="TICK: 0000  SIM TIME: --:--",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLORS["text_dim"],
        )
        self.tick_label.pack(side="right", padx=16)

        # Ollama status
        ollama_status = "● OLLAMA OK" if OLLAMA_AVAILABLE else "⚠ FALLBACK"
        ollama_color  = COLORS["green"] if OLLAMA_AVAILABLE else COLORS["yellow"]
        ctk.CTkLabel(
            bar,
            text=ollama_status,
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=ollama_color,
        ).pack(side="right", padx=12)

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTROL HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_simulation(self):
        if not self._sim_active:
            self._sim_active = True
            self.engine.start()
            self.play_btn.configure(text="⏸  PAUSE",
                                    fg_color=COLORS["orange"],
                                    hover_color=COLORS["yellow"],
                                    text_color=COLORS["bg_dark"])
            self.status_dot.configure(text="● ONLINE", text_color=COLORS["green"])
        else:
            if self.engine._paused:
                self.engine.resume()
                self.play_btn.configure(text="⏸  PAUSE",
                                        fg_color=COLORS["orange"],
                                        hover_color=COLORS["yellow"],
                                        text_color=COLORS["bg_dark"])
            else:
                self.engine.pause()
                self.play_btn.configure(text="▶  RESUME",
                                        fg_color=COLORS["green_dim"],
                                        hover_color=COLORS["green"],
                                        text_color=COLORS["bg_dark"])

    def _on_speed_change(self, value):
        val = int(value)
        self.engine.set_speed(val)
        self.speed_label.configure(text=f"{val}s/tick")

    def _on_model_change(self, value):
        self.engine.model_name = value

    # ─────────────────────────────────────────────────────────────────────────
    #  QUEUE POLLING  (GUI thread update loop)
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_queues(self):
        """
        Poll the three engine queues every 100ms from the GUI thread.
        This is the safe way to update Tkinter widgets from data produced
        by background threads (direct cross-thread widget calls crash Tkinter).
        """
        # ── Process telemetry queue ───────────────────────────────────────────
        processed = 0
        while not self.engine.telemetry_queue.empty() and processed < 8:
            payload = self.engine.telemetry_queue.get_nowait()
            self._render_telemetry(payload)
            processed += 1

        # ── Process AI result queue ───────────────────────────────────────────
        processed = 0
        while not self.engine.ai_queue.empty() and processed < 2:
            item = self.engine.ai_queue.get_nowait()
            self._render_ai_result(item)
            processed += 1

        # ── Process event log queue ───────────────────────────────────────────
        processed = 0
        while not self.engine.event_queue.empty() and processed < 20:
            msg = self.engine.event_queue.get_nowait()
            self._ai_append(msg + "\n", "event")
            processed += 1

        # ── Refresh map and tick counter ──────────────────────────────────────
        if self._sim_active:
            self._redraw_map()
            self.tick_label.configure(
                text=f"TICK: {self.engine.tick_count:04d}  "
                     f"SIM TIME: {self.engine.sim_time.strftime('%H:%M')}"
            )

        # Schedule next poll
        if self._running:
            self.after(100, self._poll_queues)

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _telem_append(self, text: str, tag: str = ""):
        """Append styled text to the telemetry textbox."""
        tb = self.telem_text
        tb.configure(state="normal")
        if tag:
            tb._textbox.insert("end", text, tag)
        else:
            tb._textbox.insert("end", text)
        tb._textbox.see("end")
        # Trim to ~600 lines for performance
        line_count = int(tb._textbox.index("end-1c").split(".")[0])
        if line_count > 600:
            tb._textbox.delete("1.0", f"{line_count-500}.0")
        tb.configure(state="disabled")

    def _ai_append(self, text: str, tag: str = ""):
        """Append styled text to the AI terminal textbox."""
        tb = self.ai_text
        tb.configure(state="normal")
        if tag:
            tb._textbox.insert("end", text, tag)
        else:
            tb._textbox.insert("end", text)
        tb._textbox.see("end")
        line_count = int(tb._textbox.index("end-1c").split(".")[0])
        if line_count > 600:
            tb._textbox.delete("1.0", f"{line_count-500}.0")
        tb.configure(state="disabled")

    def _bulletin_append(self, text: str, tag: str = ""):
        """Append styled text to the bulletin board."""
        tb = self.bulletin_text
        tb.configure(state="normal")
        if tag:
            tb._textbox.insert("end", text, tag)
        else:
            tb._textbox.insert("end", text)
        tb._textbox.see("end")
        tb.configure(state="disabled")

    def _render_telemetry(self, payload: dict):
        """Format and display a LoRa payload in the telemetry stream panel."""
        node     = payload["node_id"]
        ts       = payload["timestamp"]
        sev      = payload["severity"]
        bloom    = payload["bloom_level"]
        sev_tag  = sev.lower()

        self._telem_append(f"\\n{'─'*52}\\n", "dim")
        self._telem_append(f"  ↗ {node}  @{ts}", "header")
        self._telem_append(f"  [{sev}]\\n", sev_tag)

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

            self._telem_append(f"    {label:<12}", "label")
            self._telem_append(f"{value}\\n", v_tag)
        
        self._telem_append(f"\\n    RAW: ", "dim")
        self._telem_append(f"{payload['raw']}\\n", "dim")

    def _render_ai_result(self, item: dict):
        """Format and display an AI analysis result in the AI terminal panel."""
        node    = item["node_id"]
        ts      = item["timestamp"]
        result  = item["result"]
        action  = result.get("action", "IDLE")
        sev     = result.get("severity", "LOW")
        reason  = result.get("reasoning", "—")
        bulletin= result.get("bulletin", "—")

        sev_tag = sev.lower() if sev != "IDLE" else "dim"

        # ── AI terminal output ────────────────────────────────────────────────
        self._ai_append(f"\n{'═'*50}\n", "dim")
        self._ai_append(f"  SANA-BRAIN  {node}  @{ts}\n", "header")
        self._ai_append(f"  SEVERITY : ", "dim")
        self._ai_append(f"{sev}\n", sev_tag)
        self._ai_append(f"  ACTION   : ", "dim")
        self._ai_append(f"{action}\n", "action")
        self._ai_append(f"\n  REASONING:\n", "dim")
        self._ai_append(f"  {reason}\n", "reasoning")

        # ── Bulletin board update ─────────────────────────────────────────────
        self._bulletin_append(f"\n[{ts}] ", "timestamp")
        self._bulletin_append(f"{node} ", "node")
        self._bulletin_append(f"[{sev}]", sev_tag)
        self._bulletin_append(f"\n{bulletin}\n", sev_tag if sev in ("CRITICAL","HIGH") else "")

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITY
    # ─────────────────────────────────────────────────────────────────────────

    def _update_clock(self):
        """Update the real-world clock in the header every second."""
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.clock_label.configure(text=now)
        self.after(1000, self._update_clock)

    def _on_close(self):
        """Clean shutdown."""
        self._running = False
        self.engine.stop()
        self.after(200, self.destroy)

    # ── CAMERA PANEL ──────────────────────────────────────────────────────────

    def _build_camera_panel(self, parent):
        """Dedicated panel for the live Node Camera stream."""
        content = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"],
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"])
        content.grid(row=2, column=0, sticky="nsew", padx=(0,0), pady=(0,0))

        title_bar = ctk.CTkFrame(content, fg_color=COLORS["bg_card"],
                                 corner_radius=0, height=26)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar,
            text="  📸 LIVE NODE CAMERA  ·  STREAM #01",
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=COLORS["cyan_dim"], anchor="w",
        ).pack(side="left", fill="y")

        self.cam_label = ctk.CTkLabel(content, text="INITIALIZING STREAM...",
                                      font=ctk.CTkFont(family="Courier New", size=10),
                                      text_color=COLORS["text_dim"])
        self.cam_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Start the video thread
        self.video_url = "http://192.168.1.5:5000/video"
        self.video_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.video_thread.start()

    def _video_worker(self):
        """Background worker to fetch frames from the MJPEG stream."""
        while True:
            cap = cv2.VideoCapture(self.video_url)
            if not cap.isOpened():
                self.after(0, lambda: self.cam_label.configure(text="STREAM OFFLINE (RECONNECTING...)"))
                time.sleep(5)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and convert to PIL for CTk
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                
                # Simple aspect ratio maintenance (approx 320x180)
                img = img.resize((320, 180), Image.Resampling.LANCZOS)
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(320, 180))
                
                # Update UI in main thread
                self.after(0, lambda i=ctk_img: self.cam_label.configure(image=i, text=""))
                
                # Cap at ~20 FPS to avoid overloading GUI
                time.sleep(0.05)

            cap.release()
            self.after(0, lambda: self.cam_label.configure(image=None, text="STREAM INTERRUPTED"))
            time.sleep(2)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  SANA — Smart Autonomous Natural Agent  v1.0             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Ollama available : {'YES — LLM inference active' if OLLAMA_AVAILABLE else 'NO  — using rule-based fallback':<34}║")
    print(f"║  Target model     : {DEFAULT_MODEL:<34}║")

    # ── Startup model validation ─────────────────────────────────────────────
    if OLLAMA_AVAILABLE:
        model_ok = validate_ollama_model(DEFAULT_MODEL)
        status = "VERIFIED ✓" if model_ok else "NOT FOUND ✗ (will use fallback)"
        print(f"║  Model status     : {status:<34}║")
    else:
        print(f"║  Model status     : {'SKIPPED (no ollama)':<34}║")

    print("╚══════════════════════════════════════════════════════════╝\n")

    engine = SimulationEngine()
    app    = SANADashboard(engine)
    app.mainloop()


if __name__ == "__main__":
    main()
