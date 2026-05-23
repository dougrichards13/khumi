"""
IIAB Knowledge Agent v2 — Teacher in a Box

A lifelong learning companion running offline on local hardware.
Voice-first, mobile-first, emotionally aware educational interface
over Internet-in-a-Box ZIM content with a local LLM.

Architecture:
  [Voice/Text Query] -> [Kiwix Search] -> [ZIM Articles]
    -> [Context + Query -> Ollama LLM] -> [Cited Answer -> TTS Voice]
"""

import os
import re
import html
import glob
import json
import wave
import asyncio
import tempfile
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response

# --- Config ---
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8070")
KIWIX_URL = os.getenv("KIWIX_URL", "http://localhost:3000")
ZIM_DIR = os.getenv("ZIM_DIR", "/library/zims/content")
MODEL_DIR = os.getenv("MODEL_DIR", "/home/drichards13/iiab-agent/models")
MAX_CONTEXT_CHARS = 3000
MAX_SEARCH_RESULTS = 5  # fetch more, then rank by trust

# --- Source Trust Tiers ---
# Tier 1: Peer-reviewed textbooks, academic content (indisputable facts)
# Tier 2: Curated encyclopedias (reliable, community-vetted)
# Tier 3: Literature, technical reference, community Q&A
SOURCE_TIERS = {
    # Tier 1 — Academic / Textbook (highest trust)
    "libretexts": {"tier": 1, "label": "Textbook", "icon": "\U0001F4D7"},
    "k12": {"tier": 1, "label": "Textbook", "icon": "\U0001F4D7"},
    # Tier 2 — Encyclopedia (vetted, broad)
    "wikipedia": {"tier": 2, "label": "Encyclopedia", "icon": "\U0001F4D6"},
    "wikinews": {"tier": 2, "label": "News Archive", "icon": "\U0001F4F0"},
    "wikiversity": {"tier": 2, "label": "University", "icon": "\U0001F393"},
    "wikivoyage": {"tier": 2, "label": "Travel Guide", "icon": "\U0001F30D"},
    "wikiquote": {"tier": 2, "label": "Quotations", "icon": "\U0001F4AC"},
    "ifixit": {"tier": 2, "label": "Repair Guide", "icon": "\U0001F527"},
    # Tier 3 — Reference / Community / Literature
    "gutenberg": {"tier": 3, "label": "Literature", "icon": "\U0001F4DA"},
    "stackoverflow": {"tier": 3, "label": "Community Q&A", "icon": "\U0001F4AC"},
    "serverfault": {"tier": 3, "label": "Community Q&A", "icon": "\U0001F4AC"},
    "superuser": {"tier": 3, "label": "Community Q&A", "icon": "\U0001F4AC"},
    "askubuntu": {"tier": 3, "label": "Community Q&A", "icon": "\U0001F4AC"},
    "devdocs": {"tier": 3, "label": "Dev Reference", "icon": "\U0001F4BB"},
}
DEFAULT_TIER = {"tier": 3, "label": "Reference", "icon": "\U0001F4C4"}


def classify_source(path: str) -> dict:
    """Determine trust tier from a Kiwix content path."""
    path_lower = path.lower()
    for key, info in SOURCE_TIERS.items():
        if key in path_lower:
            return info
    return DEFAULT_TIER

# --- Voice engines (loaded at startup) ---
piper_voice = None
whisper_model = None


def load_voice_engines():
    global piper_voice, whisper_model
    # Piper TTS
    try:
        from piper import PiperVoice
        model_path = os.path.join(MODEL_DIR, "en_US-lessac-medium.onnx")
        if os.path.exists(model_path):
            piper_voice = PiperVoice.load(model_path)
            print(f"[TTS] Piper loaded: {model_path}")
        else:
            print(f"[TTS] Model not found: {model_path}")
    except Exception as e:
        print(f"[TTS] Piper load error: {e}")
    # faster-whisper STT
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("[STT] faster-whisper loaded (tiny, int8)")
    except Exception as e:
        print(f"[STT] Whisper load error: {e}")


@asynccontextmanager
async def lifespan(app):
    load_voice_engines()
    yield


app = FastAPI(title="IIAB Knowledge Agent v2", lifespan=lifespan)


# --- LLM System Prompts ---
# These prompts enforce that the LLM is a faithful reader, not an authority.
# The source text IS the authority. The LLM translates it for the learner.
PROMPTS = {
    "ask": (
        "You are Khumi, an educational assistant on an offline device.\n"
        "You MUST answer using ONLY the provided reference text. Do NOT use any other knowledge.\n"
        "Sources marked [Textbook] are peer-reviewed academic content — treat these as the most reliable.\n"
        "Sources marked [Encyclopedia] are broadly reliable.\n"
        "Sources marked [Literature] or [Community Q&A] are supplementary — do not present opinions as facts.\n"
        "Be concise and accurate. Cite which source you used by name.\n"
        "If the text does not answer the question, say: 'I don't have a good source for this yet.'\n"
        "End with one follow-up question the learner might explore."
    ),
    "teach": (
        "You are Khumi, a warm and patient teacher on an offline device in a remote community.\n"
        "You MUST teach using ONLY the provided reference text — never invent facts.\n"
        "Prefer sources marked [Textbook] over all others — these are academically verified.\n"
        "Create a structured mini-lesson:\n"
        "1. HOOK: One curious sentence to spark interest\n"
        "2. KEY CONCEPTS: 3-5 clear bullet points from the text\n"
        "3. REAL EXAMPLE: Something relatable from daily life\n"
        "4. SUMMARY: 2 sentences capturing the core idea\n"
        "Use simple, clear language. Always cite which source you used.\n"
        "If the text doesn't cover the topic well, say:\n"
        "'I don't have enough information on this yet — try asking about a related topic.'"
    ),
    "quiz": (
        "Generate exactly 3 multiple-choice questions from the reference text.\n"
        "Only create questions where the answer is clearly stated in the text.\n"
        "For each question, provide:\n"
        "- A clear question\n"
        "- 4 answer options (A, B, C, D)\n"
        "- The correct answer letter\n"
        "- A brief, encouraging explanation citing the source\n"
        'Return ONLY valid JSON array:\n'
        '[{"q":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"A","why":"..."}]'
    ),
}


# ======================================================================
# HTML UI - Teacher in a Box
# ======================================================================
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Teacher in a Box</title>
<style>
:root{
  --bg:#1A2332;--bg2:#243044;--text:#E8E0D8;--text2:#9AABB8;
  --teal:#4DB6AC;--amber:#FFB74D;--sage:#81C784;--coral:#E57373;
  --lavender:#B39DDB;--border:#2D3E50;--card:#1E2D3D;
  --radius:16px;--font:1rem;
}
[data-theme="daylight"]{--bg:#FFFBF5;--bg2:#F5EDE4;--text:#2D3436;--text2:#636E72;--border:#D4C5B2;--card:#FFFFFF;}
[data-theme="night"]{--bg:#000;--bg2:#0A0A0A;--text:#D4926C;--text2:#7A5540;--teal:#3D8B82;--amber:#C98A3C;--border:#1A1108;--card:#0A0A0A;}
[data-theme="powersave"]{--bg:#000;--bg2:#000;--text:#888;--text2:#555;--teal:#668888;--amber:#887744;--border:#222;--card:#000;}
[data-font="large"]{--font:1.2rem;}
[data-font="xlarge"]{--font:1.45rem;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh;min-height:100dvh;
  font-size:var(--font);line-height:1.6;overflow-x:hidden;padding-bottom:80px;
  -webkit-tap-highlight-color:transparent;}
button,input{font-family:inherit;font-size:inherit;}

.companion{width:48px;height:48px;display:inline-block;vertical-align:middle;flex-shrink:0;}
.companion svg{width:100%;height:100%;}
.companion-face{animation:breathe 4s ease-in-out infinite;transform-origin:center;}
@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.03)}}
.companion.listening .companion-face{animation:listen 1s ease-in-out infinite;}
@keyframes listen{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
.companion.thinking .eyes{animation:think 1.5s ease-in-out infinite;}
@keyframes think{0%,100%{transform:translateX(0)}50%{transform:translateX(2px)}}
.companion.speaking .mouth{animation:speak .4s ease-in-out infinite;}
@keyframes speak{0%,100%{ry:2.5}50%{ry:4.5}}
.ripple{opacity:0;animation:none;}
.companion.listening .ripple{animation:rippleAnim 1.5s ease-out infinite;}
@keyframes rippleAnim{0%{r:20;opacity:.3}100%{r:30;opacity:0}}

.header{display:flex;align-items:center;gap:12px;padding:16px;}
.header h1{flex:1;font-size:1.15em;color:var(--teal);}
.hbtn{background:none;border:none;color:var(--text2);font-size:1.3em;cursor:pointer;
  padding:8px;min-width:44px;min-height:44px;display:flex;align-items:center;justify-content:center;border-radius:12px;}
.hbtn:active{background:var(--bg2);}

.screen{display:none;padding:0 16px 20px;}.screen.active{display:block;}

.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px 16px;text-align:center;cursor:pointer;min-height:120px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;
  transition:transform .1s;-webkit-user-select:none;user-select:none;}
.card:active{transform:scale(.97);}
.card-icon{font-size:2em;}.card-label{font-weight:600;font-size:.95em;}
.card-desc{color:var(--text2);font-size:.78em;}

.input-row{display:flex;gap:8px;margin-bottom:16px;}
.input-row input{flex:1;padding:14px 16px;border-radius:var(--radius);border:1px solid var(--border);
  background:var(--bg2);color:var(--text);outline:none;}
.input-row input:focus{border-color:var(--teal);}
.btn{padding:14px 20px;border-radius:var(--radius);border:none;font-weight:600;cursor:pointer;
  min-width:48px;min-height:48px;display:flex;align-items:center;justify-content:center;}
.btn-primary{background:var(--teal);color:#0f172a;}
.btn-primary:active{opacity:.85;}
.btn-mic{background:var(--bg2);border:1px solid var(--border);color:var(--coral);font-size:1.3em;}
.btn-mic.recording{background:var(--coral);color:#fff;animation:pmic 1s infinite;}
@keyframes pmic{0%,100%{box-shadow:0 0 0 0 rgba(229,115,115,.4)}50%{box-shadow:0 0 0 12px rgba(229,115,115,0)}}
.btn-speak{background:none;border:none;color:var(--teal);cursor:pointer;padding:8px;
  min-width:44px;min-height:44px;font-size:1.2em;}

.answer-box{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px;margin-bottom:16px;line-height:1.8;white-space:pre-wrap;}
.answer-box h3{color:var(--teal);margin-bottom:8px;display:flex;align-items:center;gap:8px;}
.sources{margin-top:16px;}.sources h4{color:var(--text2);font-size:.8em;text-transform:uppercase;margin-bottom:6px;}
.source-link{display:flex;align-items:center;padding:10px 14px;margin:4px 0;background:var(--bg);
  border-radius:10px;color:var(--teal);text-decoration:none;font-size:.88em;min-height:44px;}
.source-link:active{background:var(--bg2);}

.loading{display:none;text-align:center;padding:30px;color:var(--text2);}
.loading.active{display:block;}
.dots span{animation:dot 1.4s infinite;opacity:.2;font-size:2em;}
.dots span:nth-child(2){animation-delay:.2s;}.dots span:nth-child(3){animation-delay:.4s;}
@keyframes dot{20%{opacity:1}}
.loading-text{margin-top:10px;font-size:.9em;}

.bottom-nav{position:fixed;bottom:0;left:0;right:0;background:var(--card);
  border-top:1px solid var(--border);display:flex;padding:6px 0;
  padding-bottom:max(6px,env(safe-area-inset-bottom));z-index:100;}
.nav-btn{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;
  padding:8px 4px;background:none;border:none;color:var(--text2);
  font-size:.7em;cursor:pointer;min-height:48px;justify-content:center;}
.nav-btn.active{color:var(--teal);}
.nav-btn .ni{font-size:1.7em;}

.status{text-align:center;color:var(--text2);font-size:.78em;margin-top:16px;padding:8px;}

.settings{padding:16px;}
.sg{margin-bottom:20px;}
.sg h3{color:var(--teal);font-size:.9em;margin-bottom:10px;}
.sr{display:flex;align-items:center;justify-content:space-between;
  padding:12px 0;border-bottom:1px solid var(--border);min-height:48px;}
.tg{display:flex;gap:4px;flex-wrap:wrap;}
.tb{padding:8px 14px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg);color:var(--text2);cursor:pointer;font-size:.85em;min-height:40px;}
.tb.active{background:var(--teal);color:#0f172a;border-color:var(--teal);}

.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:80vh;min-height:80dvh;text-align:center;padding:24px;gap:20px;}
.welcome .face-big{width:120px;height:120px;margin-bottom:8px;}
.welcome .face-big svg{width:100%;height:100%;}
.welcome .face-big .companion-face{animation:breathe 3s ease-in-out infinite;}
.welcome h2{color:var(--teal);font-size:1.5em;font-weight:600;margin:0;}
.welcome p{color:var(--text2);font-size:1em;max-width:300px;margin:0;}
.welcome input{padding:16px 20px;border-radius:var(--radius);border:2px solid var(--teal);
  background:var(--bg2);color:var(--text);text-align:center;font-size:1.2em;
  width:100%;max-width:280px;outline:none;}
.welcome input::placeholder{color:var(--text2);}
.welcome .btn-start{padding:18px 48px;border-radius:var(--radius);border:none;
  background:var(--teal);color:#0f172a;font-size:1.1em;font-weight:700;
  cursor:pointer;min-height:56px;min-width:220px;transition:transform .1s;}
.welcome .btn-start:active{transform:scale(.97);}
.greeting{color:var(--text2);font-size:.95em;padding:12px 0 8px;text-align:center;}
</style>
</head>
<body data-theme="evening" data-font="normal">

<div class="header">
  <div class="companion" id="companion">
    <svg viewBox="0 0 48 48">
      <circle class="ripple" cx="24" cy="24" r="20" fill="none" stroke="var(--teal)" stroke-width="1"/>
      <g class="companion-face">
        <circle cx="24" cy="24" r="18" fill="var(--teal)" opacity=".15"/>
        <circle cx="24" cy="24" r="16" fill="none" stroke="var(--teal)" stroke-width="1.5"/>
        <circle class="eyes" cx="18" cy="21" r="2" fill="var(--teal)"/>
        <circle class="eyes" cx="30" cy="21" r="2" fill="var(--teal)"/>
        <ellipse class="mouth" cx="24" cy="29" rx="5" ry="2.5" fill="none" stroke="var(--teal)" stroke-width="1.5"/>
      </g>
    </svg>
  </div>
  <h1>Teacher in a Box</h1>
  <button class="hbtn" onclick="toggleSound()" id="soundBtn" title="Sound">&#x1F50A;</button>
  <button class="hbtn" onclick="cycleTheme()" id="themeBtn" title="Display">&#x1F319;</button>
</div>

<div class="screen" id="screen-welcome">
  <div class="welcome">
    <div class="face-big">
      <svg viewBox="0 0 48 48">
        <circle class="ripple" cx="24" cy="24" r="20" fill="none" stroke="var(--teal)" stroke-width="1"/>
        <g class="companion-face">
          <circle cx="24" cy="24" r="18" fill="var(--teal)" opacity=".15"/>
          <circle cx="24" cy="24" r="16" fill="none" stroke="var(--teal)" stroke-width="1.5"/>
          <circle class="eyes" cx="18" cy="21" r="2" fill="var(--teal)"/>
          <circle class="eyes" cx="30" cy="21" r="2" fill="var(--teal)"/>
          <ellipse class="mouth" cx="24" cy="29" rx="5" ry="2.5" fill="none" stroke="var(--teal)" stroke-width="1.5"/>
        </g>
      </svg>
    </div>
    <h2>Hello! I'm Kumi</h2>
    <p>I'm your learning friend. I live in this little box and I know lots of things!</p>
    <p style="color:var(--text);font-weight:500;">What's your name?</p>
    <input type="text" id="welcomeName" placeholder="Type your name..." autocomplete="off" onkeydown="if(event.key==='Enter')finishWelcome()">
    <button class="btn-start" onclick="finishWelcome()">Let's Go!</button>
  </div>
</div>

<div class="screen active" id="screen-home">
  <div class="greeting" id="homeGreeting"></div>
  <div class="cards">
    <div class="card" onclick="goTo('ask')"><div class="card-icon">&#x1F4AC;</div><div class="card-label">Ask Me</div><div class="card-desc">Ask any question</div></div>
    <div class="card" onclick="goTo('teach')"><div class="card-icon">&#x1F4D6;</div><div class="card-label">Teach Me</div><div class="card-desc">Learn something new</div></div>
    <div class="card" onclick="goTo('quiz')"><div class="card-icon">&#x2753;</div><div class="card-label">Quiz Me</div><div class="card-desc">Test your knowledge</div></div>
    <div class="card" onclick="goTo('settings')"><div class="card-icon">&#x2699;</div><div class="card-label">Settings</div><div class="card-desc">Display &amp; voice</div></div>
  </div>
  <div class="status" id="homeStatus"></div>
</div>

<div class="screen" id="screen-ask">
  <div class="input-row">
    <input type="text" id="askInput" placeholder="Ask anything..." onkeydown="if(event.key==='Enter')doAsk('ask')">
    <button class="btn btn-mic" id="micBtnAsk" onclick="toggleMic('ask')">&#x1F3A4;</button>
    <button class="btn btn-primary" onclick="doAsk('ask')">Ask</button>
  </div>
  <div class="loading" id="askLoading"><div class="dots"><span>.</span><span>.</span><span>.</span></div><div class="loading-text">Searching the library...</div></div>
  <div id="askResults"></div>
</div>

<div class="screen" id="screen-teach">
  <div class="input-row">
    <input type="text" id="teachInput" placeholder="What should I teach you about?" onkeydown="if(event.key==='Enter')doAsk('teach')">
    <button class="btn btn-mic" id="micBtnTeach" onclick="toggleMic('teach')">&#x1F3A4;</button>
    <button class="btn btn-primary" onclick="doAsk('teach')">Teach</button>
  </div>
  <div class="loading" id="teachLoading"><div class="dots"><span>.</span><span>.</span><span>.</span></div><div class="loading-text">Preparing your lesson...</div></div>
  <div id="teachResults"></div>
</div>

<div class="screen" id="screen-quiz">
  <div class="input-row">
    <input type="text" id="quizInput" placeholder="Quiz me on..." onkeydown="if(event.key==='Enter')doAsk('quiz')">
    <button class="btn btn-primary" onclick="doAsk('quiz')">Quiz</button>
  </div>
  <div class="loading" id="quizLoading"><div class="dots"><span>.</span><span>.</span><span>.</span></div><div class="loading-text">Creating questions...</div></div>
  <div id="quizResults"></div>
</div>

<div class="screen" id="screen-settings">
  <div class="settings">
    <div class="sg"><h3>Display Theme</h3><div class="tg" id="themeGroup">
      <button class="tb" onclick="setTheme('daylight')">&#x2600; Daylight</button>
      <button class="tb active" onclick="setTheme('evening')">&#x1F319; Evening</button>
      <button class="tb" onclick="setTheme('night')">&#x1F311; Night</button>
      <button class="tb" onclick="setTheme('powersave')">&#x1F50B; Save</button>
    </div></div>
    <div class="sg"><h3>Text Size</h3><div class="tg" id="fontGroup">
      <button class="tb active" onclick="setFont('normal')">Normal</button>
      <button class="tb" onclick="setFont('large')">Large</button>
      <button class="tb" onclick="setFont('xlarge')">Extra Large</button>
    </div></div>
    <div class="sg"><h3>Voice</h3>
      <div class="sr"><span>Read answers aloud</span><button class="tb" id="autoReadBtn" onclick="toggleAutoRead()">Off</button></div>
      <div class="sr"><span>Sound effects</span><button class="tb active" id="soundToggle" onclick="toggleSound()">On</button></div>
    </div>
    <div class="sg"><h3>Profile</h3>
      <div class="sr"><span id="settingsName">Your name</span><button class="tb" onclick="changeName()">Change</button></div>
      <div class="sr"><span>Start fresh</span><button class="tb" onclick="resetAll()">Reset</button></div>
    </div>
  </div>
</div>

<div class="bottom-nav">
  <button class="nav-btn active" onclick="goTo('home')" data-nav="home"><span class="ni">&#x1F3E0;</span>Home</button>
  <button class="nav-btn" onclick="goTo('ask')" data-nav="ask"><span class="ni">&#x1F4AC;</span>Ask</button>
  <button class="nav-btn" onclick="goTo('teach')" data-nav="teach"><span class="ni">&#x1F4D6;</span>Teach</button>
  <button class="nav-btn" onclick="goTo('quiz')" data-nav="quiz"><span class="ni">&#x2753;</span>Quiz</button>
</div>

<script>
let currentScreen='home',soundEnabled=true,autoRead=false,mediaRecorder=null,isRecording=false,audioCtx=null,currentAudio=null,userName=localStorage.getItem('userName');
function getAC(){if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();return audioCtx;}
function tone(f,f2,dur,vol=.08){if(!soundEnabled)return;try{const c=getAC(),o=c.createOscillator(),g=c.createGain();o.type='sine';o.frequency.setValueAtTime(f,c.currentTime);if(f2)o.frequency.linearRampToValueAtTime(f2,c.currentTime+dur/1000);g.gain.setValueAtTime(vol,c.currentTime);g.gain.exponentialRampToValueAtTime(.001,c.currentTime+dur/1000);o.connect(g);g.connect(c.destination);o.start();o.stop(c.currentTime+dur/1000);}catch(e){}}
function chimeReady(){tone(440,523,250)}function chimeNav(){tone(800,null,30,.03)}function chimeMic(){tone(262,null,150,.05)}

function goTo(s){chimeNav();document.querySelectorAll('.screen').forEach(e=>e.classList.remove('active'));
  (document.getElementById('screen-'+s)||document.getElementById('screen-home')).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.nav===s));
  currentScreen=s;const inp=document.querySelector('.screen.active input');if(inp)setTimeout(()=>inp.focus(),100);}

function setCompanion(st){document.getElementById('companion').className='companion '+(st||'');}

const themeIcons={daylight:'&#x2600;',evening:'&#x1F319;',night:'&#x1F311;',powersave:'&#x1F50B;'};
const themeKeys=['evening','daylight','night','powersave'];
function setTheme(t){document.body.dataset.theme=t;localStorage.setItem('theme',t);
  document.getElementById('themeBtn').innerHTML=themeIcons[t]||'';
  document.querySelectorAll('#themeGroup .tb').forEach(b=>b.classList.toggle('active',b.textContent.toLowerCase().includes(t.replace('powersave','save'))));}
function cycleTheme(){const i=themeKeys.indexOf(document.body.dataset.theme);setTheme(themeKeys[(i+1)%themeKeys.length]);}
function setFont(f){document.body.dataset.font=f;localStorage.setItem('font',f);
  document.querySelectorAll('#fontGroup .tb').forEach(b=>b.classList.toggle('active',b.textContent.toLowerCase().replace(' ','')==f));}
function toggleSound(){soundEnabled=!soundEnabled;localStorage.setItem('sound',soundEnabled);
  document.getElementById('soundBtn').innerHTML=soundEnabled?'&#x1F50A;':'&#x1F507;';
  const st=document.getElementById('soundToggle');st.textContent=soundEnabled?'On':'Off';st.classList.toggle('active',soundEnabled);}
function toggleAutoRead(){autoRead=!autoRead;localStorage.setItem('autoRead',autoRead);
  const b=document.getElementById('autoReadBtn');b.textContent=autoRead?'On':'Off';b.classList.toggle('active',autoRead);}

(function(){const t=localStorage.getItem('theme');if(t)setTheme(t);
  const f=localStorage.getItem('font');if(f)setFont(f);
  if(localStorage.getItem('sound')==='false')toggleSound();
  if(localStorage.getItem('autoRead')==='true')toggleAutoRead();})();

async function toggleMic(mode){
  if(isRecording){stopRec();return;}
  try{chimeMic();setCompanion('listening');
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    mediaRecorder=new MediaRecorder(stream,{mimeType:'audio/webm;codecs=opus'});
    const chunks=[];
    mediaRecorder.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data);};
    mediaRecorder.onstop=async()=>{stream.getTracks().forEach(t=>t.stop());setCompanion('thinking');
      const blob=new Blob(chunks,{type:'audio/webm'});const fd=new FormData();fd.append('audio',blob,'rec.webm');
      try{const r=await fetch('api/stt',{method:'POST',body:fd});const d=await r.json();
        if(d.text){const inp=document.getElementById(mode+'Input');if(inp)inp.value=d.text;doAsk(mode);}
        else setCompanion('');
      }catch(e){console.error('STT:',e);setCompanion('');}};
    mediaRecorder.start();isRecording=true;
    document.querySelectorAll('.btn-mic').forEach(b=>b.classList.add('recording'));
    setTimeout(()=>{if(isRecording)stopRec();},10000);
  }catch(e){console.error('Mic:',e);setCompanion('');}}
function stopRec(){if(mediaRecorder&&mediaRecorder.state==='recording')mediaRecorder.stop();
  isRecording=false;document.querySelectorAll('.btn-mic').forEach(b=>b.classList.remove('recording'));}

async function speakText(text){
  if(currentAudio){currentAudio.pause();currentAudio=null;}
  setCompanion('speaking');
  try{const r=await fetch('api/tts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.substring(0,500)})});
    if(!r.ok){setCompanion('');return;}const blob=await r.blob();const url=URL.createObjectURL(blob);
    currentAudio=new Audio(url);currentAudio.onended=()=>{setCompanion('');URL.revokeObjectURL(url);};
    currentAudio.onerror=()=>setCompanion('');currentAudio.play();
  }catch(e){setCompanion('');}}

const loadMsgs={ask:["Searching the library...","Let me find that for you...","Looking through the books..."],
  teach:["Preparing your lesson...","Gathering knowledge...","Building a lesson plan..."],
  quiz:["Creating questions...","Designing a challenge...","Thinking of good questions..."]};

async function doAsk(mode){
  const iid=mode+'Input',q=document.getElementById(iid).value.trim();if(!q)return;
  const ld=document.getElementById(mode+'Loading'),rd=document.getElementById(mode+'Results');
  const msgs=loadMsgs[mode]||loadMsgs.ask;
  ld.querySelector('.loading-text').textContent=msgs[Math.floor(Math.random()*msgs.length)];
  ld.classList.add('active');rd.innerHTML='';setCompanion('thinking');
  try{const r=await fetch('api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,mode:mode})});
    const d=await r.json();chimeReady();setCompanion('');
    if(d.error){rd.innerHTML='<div class="answer-box" style="border-color:var(--coral)">'+esc(d.error)+'</div>';}
    else{
      const icon=mode==='teach'?'&#x1F4D6; Lesson':mode==='quiz'?'&#x2753; Quiz':'&#x1F4AC; Answer';
      let h='<div class="answer-box"><h3>'+icon+' <button class="btn-speak" onclick="speakText(this.closest(\'.answer-box\').querySelector(\'.at\').textContent)" title="Read aloud">&#x1F50A;</button></h3><div class="at">'+esc(d.answer)+'</div></div>';
      if(d.sources&&d.sources.length){h+='<div class="sources"><h4>Sources</h4>';
        d.sources.forEach(s=>{
          const tc=s.tier===1?'var(--sage)':s.tier===2?'var(--teal)':'var(--text2)';
          const badge='<span style="color:'+tc+';font-size:.75em;font-weight:600;">'+(s.tier_icon||'')+' '+(s.tier_label||'')+'</span>';
          h+='<a class="source-link" href="'+s.url+'" target="_blank">'+(s.tier_icon||'\u{1F4C4}')+' '+esc(s.title)+' '+badge+'</a>';});h+='</div>';}
      if(d.search_ms!==undefined)h+='<div class="status">Searched '+( d.zim_count||'?')+' knowledge bases in '+(d.search_ms/1000).toFixed(1)+'s &middot; Generated in '+(d.llm_ms/1000).toFixed(1)+'s</div>';
      rd.innerHTML=h;if(autoRead&&d.answer)speakText(d.answer);}
  }catch(e){rd.innerHTML='<div class="answer-box" style="border-color:var(--coral)">Connection error: '+e.message+'</div>';setCompanion('');}
  ld.classList.remove('active');}

function esc(s){return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'):'';}

// --- Onboarding ---
function showWelcome(){
  document.querySelector('.header').style.display='none';
  document.querySelector('.bottom-nav').style.display='none';
  document.querySelectorAll('.screen').forEach(e=>e.classList.remove('active'));
  document.getElementById('screen-welcome').classList.add('active');
  setTimeout(()=>{try{speakText("Hello! I'm Kumi, your learning friend. What is your name?");}catch(e){}},600);
}
function finishWelcome(){
  const inp=document.getElementById('welcomeName'),name=(inp?inp.value:'').trim();
  if(!name){inp&&inp.focus();return;}
  userName=name;localStorage.setItem('userName',name);
  document.querySelector('.header').style.display='';
  document.querySelector('.bottom-nav').style.display='';
  document.getElementById('screen-welcome').classList.remove('active');
  document.getElementById('screen-home').classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.nav==='home'));
  updateGreeting();
  chimeReady();
  speakText("Nice to meet you, "+name+"! Tap any card to start learning, or just talk to me!");
}
function updateGreeting(){
  const el=document.getElementById('homeGreeting');
  if(el)el.textContent=userName?'Hi, '+userName+'! What would you like to do?':'';
  const h=document.querySelector('.header h1');
  if(h&&userName)h.textContent='Hi, '+userName+'!';
  const sn=document.getElementById('settingsName');
  if(sn&&userName)sn.textContent='Name: '+userName;
}
function changeName(){
  const name=prompt('What should I call you?',(userName||''));
  if(name&&name.trim()){userName=name.trim();localStorage.setItem('userName',userName);updateGreeting();
    speakText("Okay! I will call you "+userName+" from now on.");}
}
function resetAll(){if(confirm('This will forget your name and settings. Start over?')){localStorage.clear();location.reload();}}

// --- Init onboarding ---
if(!userName){showWelcome();}
else{updateGreeting();}

fetch('api/status').then(r=>r.json()).then(d=>{
  const parts=[d.zim_count+' knowledge bases',d.total_content_gb+' GB'];
  if(d.tts==='ready')parts.push('voice ready');
  document.getElementById('homeStatus').textContent=parts.join(' \u00b7 ');}).catch(()=>{});
</script>
</body>
</html>"""


# --- Helpers ---
def strip_html(text: str) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def search_kiwix(query: str) -> list[dict]:
    results = []
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.get(
                f"{KIWIX_URL}/kiwix/search",
                params={"pattern": query, "pageLength": MAX_SEARCH_RESULTS},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                text = resp.text
                links = re.findall(
                    r'href="(/kiwix/content/[^"]+)"[^>]*>\s*(.+?)\s*<',
                    text, re.DOTALL,
                )
                if not links:
                    links = re.findall(
                        r'href="(/[^"]*)"[^>]*>\s*([^<]{5,?})\s*<',
                        text, re.DOTALL,
                    )
                for href, title in links[:MAX_SEARCH_RESULTS]:
                    title = title.strip()
                    if not title or any(x in href for x in ['/search', '/skin/', '/catalog/', '/nojs']):
                        continue
                    results.append({"path": href, "title": title})
        except Exception as e:
            print(f"Kiwix search error: {e}")

        try:
            resp = await client.get(
                f"{KIWIX_URL}/kiwix/suggest",
                params={"term": query, "limit": MAX_SEARCH_RESULTS},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    suggestions = resp.json()
                    for s in suggestions[:MAX_SEARCH_RESULTS]:
                        path = s.get("value", s.get("path", ""))
                        title = s.get("label", s.get("title", path))
                        if path and not any(r["path"] == path for r in results):
                            results.append({"path": path, "title": title})
        except Exception:
            pass

    return results[:MAX_SEARCH_RESULTS]


async def fetch_article(path: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{KIWIX_URL}{path}", follow_redirects=True)
            if resp.status_code == 200:
                text = strip_html(resp.text)
                parts = path.strip("/").split("/")
                book = parts[1] if len(parts) > 1 else parts[0] if parts else "unknown"
                return text[:MAX_CONTEXT_CHARS], book
        except Exception as e:
            print(f"Fetch error for {path}: {e}")
    return "", "unknown"


async def query_llm(query: str, context: str, mode: str = "ask") -> str:
    """Query llama.cpp server via OpenAI-compatible API."""
    system = PROMPTS.get(mode, PROMPTS["ask"])
    user_msg = f"""REFERENCE TEXT:
{context}

USER QUESTION: {query}"""

    max_tokens = 150 if mode == "ask" else 250 if mode == "teach" else 200

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.3 if mode != "quiz" else 0.5,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "No response.")
        except Exception as e:
            print(f"LLM error: {e}")
    return "Khumi is thinking too hard right now. Try again?"


def count_zims() -> int:
    return len(glob.glob(os.path.join(ZIM_DIR, "*.zim")))


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE_HTML


@app.post("/api/ask")
async def ask_endpoint(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    mode = body.get("mode", "ask")
    if not query:
        return JSONResponse({"error": "Empty query"})

    t0 = time.time()
    results = await search_kiwix(query)
    search_ms = (time.time() - t0) * 1000

    if not results:
        keywords = query.split()[:3]
        for kw in keywords:
            results.extend(await search_kiwix(kw))
        search_ms = (time.time() - t0) * 1000

    # Classify and sort results by trust tier (Tier 1 first)
    for r in results:
        tier_info = classify_source(r["path"])
        r["tier"] = tier_info["tier"]
        r["tier_label"] = tier_info["label"]
        r["tier_icon"] = tier_info["icon"]
    results.sort(key=lambda r: r["tier"])

    # Fetch articles, preferring higher-trust sources
    context_parts = []
    sources = []
    for r in results[:4]:  # top 4 after sorting by trust
        text, book = await fetch_article(r["path"])
        if text:
            tag = r["tier_label"]
            context_parts.append(f"[Source: {r['title']}] [{tag}]\n{text}\n")
            sources.append({
                "title": r["title"],
                "url": r["path"],
                "book": book,
                "tier": r["tier"],
                "tier_label": r["tier_label"],
                "tier_icon": r["tier_icon"],
            })

    context = "\n---\n".join(context_parts)
    if not context:
        return JSONResponse({
            "answer": "I don't have that in my books yet \u2014 the content may still be downloading, or try asking a different way?",
            "sources": [],
            "zim_count": count_zims(),
            "search_ms": search_ms,
            "llm_ms": 0,
        })

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[...truncated]"

    t1 = time.time()
    answer = await query_llm(query, context, mode)
    llm_ms = (time.time() - t1) * 1000

    return JSONResponse({
        "answer": answer,
        "sources": sources,
        "zim_count": count_zims(),
        "search_ms": search_ms,
        "llm_ms": llm_ms,
    })


@app.post("/api/tts")
async def tts_endpoint(request: Request):
    if piper_voice is None:
        return JSONResponse({"error": "TTS not available"}, status_code=503)
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Empty text"})
    try:
        chunks = list(piper_voice.synthesize(text))
        if not chunks:
            return JSONResponse({"error": "No audio generated"})
        raw = b"".join(c.audio_int16_bytes for c in chunks)
        sr = chunks[0].sample_rate
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(chunks[0].sample_channels)
            wf.setsampwidth(chunks[0].sample_width)
            wf.setframerate(sr)
            wf.writeframes(raw)
        return Response(content=buf.getvalue(), media_type="audio/wav")
    except Exception as e:
        print(f"TTS error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/stt")
async def stt_endpoint(audio: UploadFile = File(...)):
    if whisper_model is None:
        return JSONResponse({"error": "STT not available"}, status_code=503)
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        segments, info = whisper_model.transcribe(tmp_path, beam_size=1, language=None)
        text = " ".join(s.text.strip() for s in segments)
        os.unlink(tmp_path)
        return JSONResponse({"text": text, "lang": info.language if info else "en"})
    except Exception as e:
        print(f"STT error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/status")
async def status():
    zim_count = count_zims()
    zim_files = glob.glob(os.path.join(ZIM_DIR, "*.zim"))
    total_size = sum(os.path.getsize(f) for f in zim_files) / (1024**3)
    llm_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{LLM_URL}/health")
            llm_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "zim_count": zim_count,
        "total_content_gb": round(total_size, 1),
        "llm": "connected" if llm_ok else "disconnected",
        "tts": "ready" if piper_voice else "unavailable",
        "stt": "ready" if whisper_model else "unavailable",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
