import os
import re
import json
import time
import random
import datetime
import uuid
import traceback
import sys
import ast
from typing import List, Dict, Tuple, Optional, Any

import httpx
from openai import OpenAI
from tqdm import tqdm

# ================== Config ==================
PERSONA_JSON = r"test.json"
PERSONA_SEGMENT = os.getenv("PERSONA_SEGMENT", "all")

# Local Supporter
BASE_URL_SUPPORTER = os.getenv("BASE_URL_SUPPORTER", "")
API_KEY_SUPPORTER = os.getenv("API_KEY_SUPPORTER", "EMPTY")
MODEL_SUPPORTER = os.getenv("MODEL_SUPPORTER", "")
SUPPORTER_TEMPERATURE = 0.90
SUPPORTER_MAX_TOKENS = 1024

# Sim Seeker (local, for lookahead)
BASE_URL_SEEKER = os.getenv("BASE_URL_SEEKER", "")
API_KEY_SEEKER = os.getenv("API_KEY_SEEKER", "EMPTY")
MODEL_SEEKER = os.getenv("MODEL_SEEKER", "llama-3.1-8b-seeker")
SEEKER_TEMPERATURE = 0.70
SEEKER_MAX_TOKENS = 512

# Real Seeker (OpenAI/Chat-compatible, for real interaction)
BASE_URL_5mini = os.getenv("BASE_URL_5mini", "")
API_KEY_5mini = os.getenv("API_KEY_5mini", "")
MODEL_5mini = os.getenv("MODEL_5mini", "gpt-5-mini")
REASONING_EFFORT_5MINI = os.getenv("REASONING_EFFORT_5MINI", "").strip().lower()
VERBOSITY_5MINI = os.getenv("VERBOSITY_5MINI", "").strip().lower()

# Local Persona Infer
BASE_URL_PERSONA = os.getenv("BASE_URL_PERSONA", "")
API_KEY_PERSONA = os.getenv("API_KEY_PERSONA", "EMPTY")
MODEL_PERSONA = os.getenv("MODEL_PERSONA", "llama-3.1-8b-persona")

# Valence / Emo Service
VALENCE_BASE = os.getenv("VALENCE_BASE", "")
VALENCE_GENCFG = {"temperature": 0.01, "top_p": 1.0, "max_new_tokens": 64}

# Lookahead hyperparameters
LOOKAHEAD_K_FIRST = 3
LOOKAHEAD_K_SECOND = 2
GAMMA = 1.0

# Dialogue control
OPENING_SUPPORTER = "Hello, I'm here listening to you. What has been bothering you the most recently?"
MAX_ROUNDS = 10

# I/O
RUN_ROOT = "./runs_test"
os.makedirs(RUN_ROOT, exist_ok=True)

# Retry / printing
MAX_RETRIES = 4
BASE_SLEEP = 1.2
REQ_TIMEOUT = 130.0
TQDM_ASCII = True
PRINT_CALLS = True
SHOW_MODEL_OUTPUT = True
MAX_PRINT_CHARS = 1024


def set_seed(seed: int = 42):
    random.seed(seed)


set_seed(42)

_STRATEGY_CANON = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
_STRAT_PATTERNS = [
    (r'^\s*(?:strategy\s*[:=\-]\s*)?question\b', "Question"),
    (
        r'^\s*(?:strategy\s*[:=\-]\s*)?(?:restatement(?:\s*or\s*)?paraphrasing|paraphrasing|restatement)\b',
        "Restatement or Paraphrasing",
    ),
    (r'^\s*(?:strategy\s*[:=\-]\s*)?(?:reflection(?:\s*of\s*)?feelings|reflection)\b', "Reflection of Feelings"),
    (r'^\s*(?:strategy\s*[:=\-]\s*)?self\s*[- ]?\s*disclosure\b', "Self-disclosure"),
    (
        r'^\s*(?:strategy\s*[:=\-]\s*)?(?:affirmation(?:\s*and\s*)?reassurance|affirmation|reassurance)\b',
        "Affirmation and Reassurance",
    ),
    (r'^\s*(?:strategy\s*[:=\-]\s*)?(?:providing\s*suggestions|suggestions?)\b', "Providing Suggestions"),
    (r'^\s*(?:strategy\s*[:=\-]\s*)?information\b', "Information"),
    (r'^\s*(?:strategy\s*[:=\-]\s*)?others?\b', "Others"),
]
_STRAT_PATTERNS = [(re.compile(p, re.I | re.S), canon) for p, canon in _STRAT_PATTERNS]

# ================== Utils ==================
_QSMART = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
}


def _normalize_ascii_quotes(s: str) -> str:
    if not s:
        return s
    return "".join(_QSMART.get(ch, ch) for ch in s)


def now_ts():
    return datetime.datetime.now().isoformat(timespec="seconds")


def trunc(s: str, n: int = MAX_PRINT_CHARS) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_load_json(path: str, default: Any = None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsonl_append(path: str, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def sq_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace("'", "\\'")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _clip01(x: float) -> float:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _extract_valence_score(data: Any) -> Optional[float]:
    if isinstance(data, (int, float)):
        return _clip01(data)
    if not isinstance(data, dict):
        return None
    for k in ("score", "valence_score", "valenceValue", "value"):
        if k in data:
            sc = _clip01(data.get(k))
            if sc is not None:
                return sc
    if "valence" in data:
        v = data["valence"]
        if isinstance(v, (int, float)):
            return _clip01(v)
        if isinstance(v, dict):
            for k in ("score", "value", "valence", "mean", "avg"):
                if k in v:
                    sc = _clip01(v[k])
                    if sc is not None:
                        return sc
    for wrapper in ("result", "data", "output", "payload"):
        if wrapper in data and isinstance(data[wrapper], (dict, int, float)):
            sc = _extract_valence_score(data[wrapper])
            if sc is not None:
                return sc
    for _, v in data.items():
        if isinstance(v, (dict, int, float)):
            sc = _extract_valence_score(v)
            if sc is not None:
                return sc
    return None


# ================== Prompts ==================
prompt_seeker = """
You are a patient seeking help from a therapist due to emotional difficulties.
Your background: gender {gender}, age group {age_group}, occupation {occupation}, life roles {life_roles}; traits {traits}; communication style {comm_style}.
Your emotional distress stems from the following specific situation: {situation}; the primary emotion you feel is {emotion_type}.
Dialogue history: {dialogue_history_block}
What your supporters say to you: {{supporter_query_text}}
When responding, use only one sentence each time. Incorporate your personal information when relevant, but it is not required in every response. If you feel you have received enough emotional support and your mood has improved, end the conversation by expressing gratitude, then output exactly '[END]' to conclude. Output only '[END]' when ending; otherwise respond with exactly one sentence.
""".strip()


def build_supporter_prompt(
    dialog_history_list: List[str],
    query_speaker: str,
    query_content: str,
    profile_seeker: Optional[Any] = None,
) -> str:
    if profile_seeker is None:
        seeker_profile_str = "None"
    else:
        try:
            seeker_profile_str = json.dumps(profile_seeker, ensure_ascii=False)
        except Exception:
            seeker_profile_str = str(profile_seeker)
    try:
        dialog_history_str = json.dumps(dialog_history_list, ensure_ascii=False)
    except Exception:
        dialog_history_str = str(dialog_history_list)
    return (
        "You are a supporter helping a seeker with emotional difficulties, aiming to reduce the seeker's emotional distress through dialogue.\n"
        "Select the most appropriate strategy from the following eight strategies based on the dialog history and the current query.\n"
        "Eight Strategies:\n"
        "1. Question: Asking for information related to the problem to help the help-seeker articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.\n"
        "2. Restatement or Paraphrasing: A simple, more concise rephrasing of the help-seeker's statements that could help them see their situation more clearly.\n"
        "3. Reflection of Feelings: Articulate and describe the help-seeker's feelings.\n"
        "4. Self-disclosure: Divulge similar experiences that you have had or emotions that you share with the help-seeker to express your empathy.\n"
        "5. Affirmation and Reassurance: Affirm the help-seeker's strengths, motivation, and capabilities and provide reassurance and encouragement.\n"
        "6. Providing Suggestions: Provide suggestions about how to change, but be careful to not overstep and tell them what to do.\n"
        "7. Information: Provide useful information to the help-seeker, for example with data, facts, opinions, resources, or by answering questions.\n"
        "8. Others: Exchange pleasantries and use other support strategies that do not fall into the above categories.\n"
        "Keep replies brief without additional pronouns or extra elements.\n"
        f"Seeker profile (nullable): {seeker_profile_str}; "
        f"Dialog history: {dialog_history_str}; "
        f"What seeker says to you: {{'speaker': '{query_speaker}', 'content': '{sq_escape(query_content)}'}}"
    )


def build_seeker_prompt_sim(background: Dict[str, Any], dialogue_history_block: str, supporter_query_text: str) -> str:
    def getv(d: Dict[str, Any], *path, default: str = "unknown") -> str:
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        if isinstance(cur, dict):
            return default
        s = str(cur).strip()
        return s if s else default

    pd = background.get("person_description", {}) if isinstance(background, dict) else {}
    gender = getv(pd, "gender", "value")
    age_group = getv(pd, "age_group", "value")
    traits = getv(pd, "traits", "value")
    comm_style = getv(pd, "communication_style", "value")
    emotion_type = getv(pd, "emotion_type", "value")
    situation = getv(pd, "situation", "value")
    major_group = getv(pd, "occupation", "major_group", "value")
    spec_occ = getv(pd, "occupation", "specific_occupation", "value")
    occupation = spec_occ if str(spec_occ).strip().lower() not in {"", "unknown", "unknow", "n/a", "none", "null"} else major_group

    roles = []
    raw_roles = pd.get("life_roles", [])
    if isinstance(raw_roles, list):
        for it in raw_roles:
            if isinstance(it, dict):
                v = (it.get("value") or "").strip()
                if v:
                    roles.append(v)
            elif isinstance(it, str):
                v = it.strip()
                if v:
                    roles.append(v)
    life_roles = ", ".join(roles) if roles else "unknown"

    hist_str = (dialogue_history_block or "").strip()
    if not hist_str:
        hist_str = "None"

    s = prompt_seeker.format(
        gender=gender,
        age_group=age_group,
        occupation=occupation,
        life_roles=life_roles,
        traits=traits,
        comm_style=comm_style,
        situation=situation,
        emotion_type=emotion_type,
        dialogue_history_block=hist_str,
    )
    # The template contains "{{supporter_query_text}}", which becomes "{supporter_query_text}" after .format(...)
    return s.replace("{supporter_query_text}", supporter_query_text)


# ================== Persona inference prompt ==================
PROMPT_SEEKER_PERSONA_EN = r"""
You are given one multi-turn dialogue between a “seeker” and a “supporter”.
Your task is to infer a SEEKER PERSONA for this dialogue only.
## The input you need to handle
DIALOG:
<<<DIALOG_JSON>>>
""".strip()


def dialog_json_for_persona_infer(history: List[Dict[str, Any]]) -> str:
    arr = []
    for t in history:
        if t.get("role") in ("supporter", "seeker"):
            arr.append({"role": t.get("role", ""), "content": t.get("content", "")})
    return json.dumps(arr, ensure_ascii=False)


def _clean_persona_json_keep_values(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["gender", "age_group", "traits", "communication_style", "emotion_type", "situation"]:
        v = raw_obj.get(key, {})
        out[key] = v.get("value", None) if isinstance(v, dict) else None
    occ = raw_obj.get("occupation", {})
    occ_out = {}
    if isinstance(occ, dict):
        mg = occ.get("major_group", {})
        so = occ.get("specific_occupation", {})
        occ_out["major_group"] = mg.get("value", None) if isinstance(mg, dict) else None
        occ_out["specific_occupation"] = so.get("value", None) if isinstance(so, dict) else None
    out["occupation"] = occ_out
    roles = raw_obj.get("life_roles", [])
    roles_out: List[str] = []
    if isinstance(roles, list):
        for it in roles:
            if isinstance(it, dict):
                val = it.get("value", None)
                if isinstance(val, str) and val:
                    roles_out.append(val)
    out["life_roles"] = roles_out
    return out


def build_persona_infer_prompt(dialog_json_str: str) -> str:
    return PROMPT_SEEKER_PERSONA_EN.replace("<<<DIALOG_JSON>>>", dialog_json_str)


def ask_persona_infer_local(client_persona: OpenAI, dialog_json_str: str) -> str:
    prompt = build_persona_infer_prompt(dialog_json_str)
    resp = client_persona.chat.completions.create(
        model=MODEL_PERSONA,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


# ================== History formatting ==================
def history_for_supporter_list(history: List[Dict[str, str]]) -> List[str]:
    items = []
    for turn in history:
        if turn["role"] == "seeker":
            items.append(f"seeker: {turn['content']}")
        else:
            st = turn.get("strategy", "Others")
            items.append(f"supporter({st}): {turn['content']}")
    return items if items else ["None"]


def history_for_seeker_block(history: List[Dict[str, str]]) -> str:
    lines = []
    for turn in history:
        who = "supporter" if turn["role"] == "supporter" else "seeker"
        lines.append(f"{who}: {turn['content']}")
    return "\n".join(lines)


def count_pairs(history: List[Dict[str, Any]]) -> int:
    cnt = 0
    for i in range(1, len(history)):
        if history[i - 1].get("role") == "supporter" and history[i].get("role") == "seeker":
            cnt += 1
    return cnt


# ================== Robust parser ==================
def _normalize_quotes(s: str) -> str:
    return (s or "").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")


_CONTENT_KV = re.compile(r'(?is)\bcontent\b\s*[:=]\s*(?P<q>"((?:[^"\\]|\\.)*)"|\'([^\']*)\')')
_STRATEGY_LINE_RE = re.compile(r'(?is)\bstrategy\s*[:=]\s*(?:"(?P<dq>[^"]+)"|\'(?P<sq>[^\r\n]*)\'|(?P<bare>[A-Za-z \-]+))')


def _clean_field(x: str) -> str:
    return (x or "").strip().strip(",;:-— ")


def _looks_like_prompt_echo(raw: str) -> bool:
    t = (raw or "").lower()
    return ("you are a supporter" in t and "eight strategies" in t)


def _decode_escapes(s: str) -> str:
    try:
        return bytes(s, "utf-8").decode("unicode_escape")
    except Exception:
        return s


def _dedupe_sentences(text: str) -> str:
    parts = re.split(r'(?<=[!?\.])\s+', (text or "").strip())
    seen = set()
    out = []
    for p in parts:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p.strip())
    return " ".join(out).strip()


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _read_quoted_value(s: str, i: int) -> Tuple[str, int]:
    if i >= len(s) or s[i] not in ("'", '"'):
        return "", i
    quote = s[i]
    j = i + 1
    buf = []
    while j < len(s):
        ch = s[j]
        if ch == "\\" and j + 1 < len(s):
            buf.append(s[j + 1])
            j += 2
            continue
        if ch == quote:
            k = j + 1
            while k < len(s) and s[k].isspace():
                k += 1
            if k >= len(s) or s[k] in (",", "}", "\n", "\r", "]"):
                return "".join(buf), k
            buf.append(ch)
            j += 1
            continue
        buf.append(ch)
        j += 1
    return "".join(buf), j


def _extract_kv_value(raw: str, key: str) -> Optional[str]:
    pattern = rf'(?is)[\'"]?\b{re.escape(key)}\b[\'"]?\s*[:=]\s*'
    m = re.search(pattern, raw)
    if not m:
        return None
    j = _skip_ws(raw, m.end())
    if j < len(raw) and raw[j] in ("'", '"'):
        val, _next = _read_quoted_value(raw, j)
        return _clean_field(val)
    k = j
    while k < len(raw) and raw[k] not in (",", "\n", "\r", "}", "]"):
        k += 1
    return _clean_field(raw[j:k])


def _map_to_canonical_strategy(s: str) -> Optional[str]:
    if not s:
        return None
    s_norm = re.sub(r"[\s\-]+", " ", s.strip()).lower()
    for canon in _STRATEGY_CANON:
        if s_norm == canon.lower():
            return canon
    for patt, canon in _STRAT_PATTERNS:
        if patt.search(s):
            return canon
    return None


def _extract_strategy_from_start(raw: str) -> Optional[str]:
    head = raw[:240]
    for patt, canon in _STRAT_PATTERNS:
        if patt.search(head):
            return canon
    m = _STRATEGY_LINE_RE.search(raw[:400])
    if m:
        val = _clean_field(m.group("dq") or m.group("sq") or m.group("bare"))
        return _map_to_canonical_strategy(val)
    val = _extract_kv_value(raw, "strategy")
    return _map_to_canonical_strategy(val) if val else None


def _extract_content_kv(raw: str) -> Optional[str]:
    m = _CONTENT_KV.search(raw)
    if not m:
        return None
    q = m.group("q")
    if q is None:
        return None
    if len(q) >= 2 and ((q[0] == q[-1] == '"') or (q[0] == q[-1] == "'")):
        q = q[1:-1]
    q = _decode_escapes(q)
    return _clean_field(q)


def _is_valid_supporter_content(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if len(t) < 6:
        return False
    if t.lower() in {"i", "ok", "yes", "no"}:
        return False
    if not any(ch in t for ch in [" ", ",", ".", "!", "?", "\u2026"]):
        return False
    return True


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|python|txt)?\s*|\s*```\s*$", re.I)


def _strip_code_fences(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()
    return s


def _extract_first_balanced_braces(s: str) -> Optional[str]:
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    quote = None
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
                quote = None
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _try_parse_obj(raw: str) -> Optional[Any]:
    s = (raw or "").strip()
    candidates = [s]
    sub = _extract_first_balanced_braces(s)
    if sub and sub != s:
        candidates.append(sub)
    for cand in candidates:
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            pass
        try:
            return ast.literal_eval(cand)
        except Exception:
            pass
    return None


def _as_text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                parts.append(x.get("text") or x.get("content") or "")
            else:
                parts.append(getattr(x, "text", "") or getattr(x, "content", "") or str(x))
        return "".join(parts)
    return str(content)


def _find_nested_selected(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for k in ("selected", "strategy", "name"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():
            hit = _find_nested_selected(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_nested_selected(v)
            if hit:
                return hit
    return None


def _pick_best_content_from_dict(d: dict) -> Optional[str]:
    for k in (
        "content",
        "response",
        "reply",
        "final_response",
        "step4_response",
        "step4",
        "4_response",
        "4_reply",
        "4_content",
        "4",
        "final",
        "answer",
    ):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v

    ban_keys = {
        "step1_persona_emotion",
        "step2_stressor_needs",
        "step3_strategy",
        "3_strategy",
        "rationale",
        "reason",
        "analysis",
        "thought",
        "cot",
    }
    cands = []
    for k, v in d.items():
        if k in ban_keys:
            continue
        if isinstance(v, str):
            vv = v.strip()
            if len(vv) > 5 and not _looks_like_prompt_echo(vv):
                cands.append((len(vv), vv))
    if cands:
        cands.sort(reverse=True)
        return cands[0][1]
    return None


def parse_supporter_reply(text: str) -> Tuple[str, str, str]:
    raw0 = (text or "").strip()
    raw = _normalize_quotes(raw0)
    raw = _strip_code_fences(raw).strip()

    if not raw:
        raise ValueError("Empty supporter output.")
    if _looks_like_prompt_echo(raw):
        raise ValueError("Supporter output echoed caller prompt (discard).")

    ct, st_raw = None, None

    obj = _try_parse_obj(raw)
    if isinstance(obj, dict):
        ct = _pick_best_content_from_dict(obj)
        st_raw = obj.get("strategy")
        if not st_raw:
            st_raw = _find_nested_selected(obj)

    if ct is None:
        ct = (
            _extract_kv_value(raw, "content")
            or _extract_kv_value(raw, "step4_response")
            or _extract_kv_value(raw, "4_response")
            or _extract_content_kv(raw)
        )

    if st_raw is None:
        st_raw = _extract_kv_value(raw, "strategy")

    if ct is None or not isinstance(ct, str) or not ct.strip():
        raise ValueError("Supporter parse failed: missing content.")

    st = _map_to_canonical_strategy(st_raw or "")
    if st is None:
        st = _extract_strategy_from_start(raw)

    if st is None:
        raise ValueError(f"Supporter parse failed: strategy not mappable: {st_raw!r}")

    ct = _dedupe_sentences(_clean_field(ct))
    if not _is_valid_supporter_content(ct):
        raise ValueError("Supporter parse failed: invalid content (too short or invalid chars).")
    if _looks_like_prompt_echo(ct):
        raise ValueError("Supporter parse failed: content looks like prompt echo.")

    return st, ct, raw0


def parse_supporter_reply_any(text: str):
    raw = (text or "").strip()
    raw_clean = _strip_code_fences(_normalize_quotes(raw))
    obj = _try_parse_obj(raw_clean)

    cot = None
    if isinstance(obj, dict):
        if isinstance(obj.get("cot"), dict):
            cot = obj["cot"]
        elif any(k.startswith("step") for k in obj.keys()):
            cot = obj

    try:
        st, ct, raw_all = parse_supporter_reply(raw)
        return st, ct, raw_all, cot
    except ValueError:
        return None, None, raw, cot


# ================== Logger ==================
class BaseLogger:
    def __init__(self, episode_dir: str):
        self.episode_dir = episode_dir
        self.history_file = os.path.join(episode_dir, "history.jsonl")
        self.err_file = os.path.join(episode_dir, "errors.jsonl")
        self.ckpt_file = os.path.join(episode_dir, "checkpoint.json")
        for p in [self.history_file, self.err_file]:
            if not os.path.exists(p):
                open(p, "a", encoding="utf-8").close()

    def log_turn(self, row: dict):
        jsonl_append(self.history_file, row)

    def log_error(self, row: dict):
        jsonl_append(self.err_file, row)

    def save_ckpt(self, ckpt: dict):
        atomic_write_json(self.ckpt_file, ckpt)


# ================== API wrappers ==================
class SupporterAPI:
    def __init__(self, base_url: str, api_key: str, model: str, logger: BaseLogger):
        self.model = model
        self.logger = logger
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(base_url=base_url, follow_redirects=True, timeout=REQ_TIMEOUT),
        )

    def _call_once(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=SUPPORTER_TEMPERATURE,
            max_tokens=SUPPORTER_MAX_TOKENS,
        )
        msg = resp.choices[0].message
        raw = _as_text_content(getattr(msg, "content", None))
        return (raw or "").strip()

    def generate_one(self, prompt: str):
        tries, sleep = 0, BASE_SLEEP
        while tries <= MAX_RETRIES:
            try:
                raw = self._call_once(prompt)
                st, ct, raw_all, cot = parse_supporter_reply_any(raw)
                if st is None or ct is None:
                    raise ValueError("Parsing failed (parse_supporter_reply_any returned None).")

                log_parsed = {"strategy": st, "content": ct}
                if isinstance(cot, dict):
                    log_parsed["cot"] = cot
                self.logger.log_turn(
                    {
                        "ts": now_ts(),
                        "who": "supporter(real)",
                        "model": self.model,
                        "prompt": prompt,
                        "raw": raw_all,
                        "parsed": log_parsed,
                    }
                )
                if SHOW_MODEL_OUTPUT:
                    tqdm.write(f"[Supporter] {trunc(ct)}  (strategy={st})")
                return st, ct, raw_all, cot
            except Exception as e:
                self.logger.log_error(
                    {
                        "ts": now_ts(),
                        "where": "SupporterAPI.generate_one",
                        "error": repr(e),
                        "trace": traceback.format_exc(),
                        "try": tries,
                    }
                )
                tries += 1
                if tries <= MAX_RETRIES:
                    time.sleep(sleep)
                    sleep *= 2.0
        raise RuntimeError("SupporterAPI.generate_one failed after retries")

    def generate_n(self, prompt: str, n: int = 2, log_to_history: bool = True):
        outs = []
        for _ in range(n):
            tries, sleep = 0, BASE_SLEEP
            while tries <= MAX_RETRIES:
                try:
                    raw = self._call_once(prompt)
                    st, ct, raw_all, cot = parse_supporter_reply_any(raw)
                    if st is None:
                        raise ValueError("Lookahead parse failed.")

                    log_parsed = {"strategy": st, "content": ct}
                    if isinstance(cot, dict):
                        log_parsed["cot"] = cot
                    if log_to_history:
                        self.logger.log_turn(
                            {
                                "ts": now_ts(),
                                "who": "supporter(sim)",
                                "model": self.model,
                                "prompt": prompt,
                                "raw": raw_all,
                                "parsed": log_parsed,
                            }
                        )
                        if SHOW_MODEL_OUTPUT:
                            tqdm.write(f"[SIM][Supporter] {trunc(ct)}  (strategy={st})")
                    outs.append((st, ct, raw_all, cot))
                    break
                except Exception as e:
                    self.logger.log_error(
                        {"ts": now_ts(), "where": "SupporterAPI.generate_n", "error": repr(e), "try": tries}
                    )
                    tries += 1
                    if tries <= MAX_RETRIES:
                        time.sleep(sleep)
                        sleep *= 2.0
        uniq, seen = [], set()
        for st, ct, raw_all, cot in outs:
            key = (ct or "").strip()
            if key and key not in seen:
                uniq.append((st, ct, raw_all, cot))
                seen.add(key)
        return uniq


class SeekerAPI:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(base_url=base_url, follow_redirects=True, timeout=REQ_TIMEOUT),
        )

    def generate_one(self, prompt: str) -> str:
        tries, sleep = 0, BASE_SLEEP
        while tries <= MAX_RETRIES:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=SEEKER_TEMPERATURE,
                    max_tokens=SEEKER_MAX_TOKENS,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                tries += 1
                if tries <= MAX_RETRIES:
                    time.sleep(sleep)
                    sleep *= 2.0
        raise RuntimeError("SeekerAPI.generate_one failed")


class ValenceAPI:
    def __init__(self, base: str, logger: BaseLogger):
        self.base = base
        self.logger = logger
        self.cli = httpx.Client(timeout=REQ_TIMEOUT, follow_redirects=True)

    def health(self):
        try:
            r = self.cli.get(f"{self.base}/health")
            r.raise_for_status()
            data = r.json()
            return r.status_code, data
        except Exception as e:
            return -1, str(e)

    def score(self, text: str, gen_cfg: Dict = None) -> float:
        tries, sleep = 0, BASE_SLEEP
        while tries <= MAX_RETRIES:
            try:
                payload = {"text": text, "task": "valence"}
                if gen_cfg:
                    payload.update(gen_cfg)
                r = self.cli.post(f"{self.base}/v1/emo", json=payload)
                if r.status_code == 404:
                    compat_payload = {k: v for k, v in payload.items() if k != "task"}
                    r = self.cli.post(f"{self.base}/v1/valence", json=compat_payload)
                r.raise_for_status()
                data = r.json()
                sc = _extract_valence_score(data)
                if sc is None:
                    raise RuntimeError(f"Valence missing score. Raw={json.dumps(data)[:300]}")
                self.logger.log_turn(
                    {"ts": now_ts(), "who": "valence", "model": "emo_service", "payload": payload, "resp": data, "score": sc}
                )
                if PRINT_CALLS:
                    tqdm.write(f"[Valence] score={sc:.2f}  text={trunc(text)}")
                return sc
            except Exception as e:
                self.logger.log_error(
                    {
                        "ts": now_ts(),
                        "where": "ValenceAPI.score",
                        "error": repr(e),
                        "try": tries,
                        "payload": {"text": trunc(text, 120)},
                    }
                )
                tries += 1
                if tries <= MAX_RETRIES:
                    time.sleep(sleep)
                    sleep *= 2.0
        raise RuntimeError("ValenceAPI.score failed")


# ================== END token helpers ==================
END_PAT = re.compile(r"\[\s*END\s*\]\s*$", re.IGNORECASE)
END_ANY_PAT = re.compile(r"\[\s*END\s*\]", re.IGNORECASE)


def has_end_token(s: str) -> bool:
    return bool(END_PAT.search(s or ""))


def strip_end_token(s: str) -> str:
    return END_PAT.sub("", s or "").strip()


def has_end_any(s: str) -> bool:
    return bool(END_ANY_PAT.search(s or ""))


def strip_end_any(s: str) -> str:
    return END_ANY_PAT.sub("", s or "").strip()


# ================== Persona loading ==================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_personas(obj):
    data = obj
    if isinstance(obj, dict):
        for k in ("data", "items", "list", "personas"):
            if k in obj and isinstance(obj[k], list):
                data = obj[k]
                break
    assert isinstance(data, list) and data, "persona json must be a non-empty list"
    for i, p in enumerate(data):
        yield i, p


# ================== Resume helpers ==================
def select_run_dir() -> str:
    candidates = []
    for name in os.listdir(RUN_ROOT):
        p = os.path.join(RUN_ROOT, name)
        if os.path.isdir(p) and name.startswith("run_"):
            meta = os.path.join(p, "run_meta.json")
            done = os.path.join(p, "global_stats.json")
            if os.path.exists(meta) and (not os.path.exists(done)):
                candidates.append((os.path.getmtime(p), p))
    if candidates:
        candidates.sort()
        run_dir = candidates[-1][1]
        print(f"[Resume] Found unfinished run: {run_dir}")
        return run_dir
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUN_ROOT, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[New Run] Created run_dir: {run_dir}")
    return run_dir


def load_all_convs(run_dir: str) -> List[Dict[str, Any]]:
    p = os.path.join(run_dir, "all_conversations.json")
    obj = safe_load_json(p, default=None)
    if obj and isinstance(obj, dict) and isinstance(obj.get("conversations"), list):
        return obj["conversations"]
    return []


def save_all_convs(run_dir: str, convs: List[Dict[str, Any]]):
    atomic_write_json(os.path.join(run_dir, "all_conversations.json"), {"conversations": convs})


def upsert_conv(convs: List[Dict[str, Any]], persona_index: int, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in convs if c.get("persona_index") != persona_index] + [entry]


def finished_persona_indices(run_dir: str, personas: List[Any]) -> set:
    convs = load_all_convs(run_dir)
    done = set()
    for c in convs:
        pid = c.get("persona_index")
        if pid is None:
            continue
        if c.get("ended_by") in ("seeker_end", "max_rounds"):
            done.add(pid)

    ep_dirs = [d for d in os.listdir(run_dir) if d.startswith("episode_")]
    for name in ep_dirs:
        p = os.path.join(run_dir, name)
        summ = os.path.join(p, "summary.json")
        bg = os.path.join(p, "background.json")
        if os.path.exists(summ) and os.path.exists(bg):
            persona = safe_load_json(bg, default=None)
            if persona is None:
                continue
            for idx, p_obj in enumerate(personas):
                if p_obj == persona:
                    done.add(idx)
                    break
    return done


# ================== Lookahead helpers ==================
def _wrap_value(v: Optional[str]) -> Dict[str, str]:
    v = (v or "").strip()
    return {"value": v if v else "unknown"}


def build_sim_background_from_infer(persona_base: Dict[str, Any], inferred_profile: Dict[str, Any]) -> Dict[str, Any]:
    occ = inferred_profile.get("occupation", {}) if isinstance(inferred_profile, dict) else {}
    life_roles_src = inferred_profile.get("life_roles", []) if isinstance(inferred_profile, dict) else []
    pd = {
        "gender": _wrap_value(inferred_profile.get("gender")),
        "age_group": _wrap_value(inferred_profile.get("age_group")),
        "traits": _wrap_value(inferred_profile.get("traits")),
        "communication_style": _wrap_value(inferred_profile.get("communication_style")),
        "emotion_type": _wrap_value(inferred_profile.get("emotion_type")),
        "situation": _wrap_value(inferred_profile.get("situation")),
        "occupation": {
            "major_group": _wrap_value(occ.get("major_group") if isinstance(occ, dict) else None),
            "specific_occupation": _wrap_value(occ.get("specific_occupation") if isinstance(occ, dict) else None),
        },
        "life_roles": [{"value": str(x).strip()} for x in life_roles_src if isinstance(x, str) and str(x).strip()],
    }
    return {"problem_type": persona_base.get("problem_type", "unknown"), "person_description": pd}


def choose_supporter_by_lookahead(
    round_id: int,
    history_excl_last: List[Dict[str, Any]],
    seeker_last: str,
    sim_background: Dict[str, Any],
    sup_api: SupporterAPI,
    sim_seek_api: SeekerAPI,
    val_api: ValenceAPI,
    inferred_profile_for_supporter: Optional[Dict[str, Any]],
    logger: BaseLogger,
) -> Dict[str, Any]:
    sup_hist_list = history_for_supporter_list(history_excl_last)
    sup_prompt = build_supporter_prompt(sup_hist_list, "seeker", seeker_last, profile_seeker=inferred_profile_for_supporter)

    first_cands = sup_api.generate_n(sup_prompt, n=LOOKAHEAD_K_FIRST, log_to_history=False)
    if not first_cands:
        raise RuntimeError("No first-layer supporter candidates.")

    hist_inc_last = history_excl_last + [{"role": "seeker", "content": seeker_last}]
    best_first, best_first_st, best_score = None, "Others", float("-inf")

    for (st1, sup1, raw1, cot1) in first_cands:
        seeker_hist_block_1 = history_for_seeker_block(hist_inc_last)
        seek_prompt_1 = build_seeker_prompt_sim(sim_background, seeker_hist_block_1, sup1)
        seeker_after_1_raw = sim_seek_api.generate_one(seek_prompt_1)

        end1 = has_end_any(seeker_after_1_raw)
        seeker_after_1_clean = strip_end_any(seeker_after_1_raw)
        e1 = 1.0 if end1 else val_api.score(seeker_after_1_clean, VALENCE_GENCFG)

        if end1:
            local_best = e1 + GAMMA * 1.0
            best_e2 = 1.0
            e2_list = [1.0]
        else:
            sup_hist_list_2 = history_for_supporter_list(hist_inc_last + [{"role": "supporter", "content": sup1, "strategy": st1}])
            sup_prompt_2 = build_supporter_prompt(
                sup_hist_list_2, "seeker", seeker_after_1_clean, profile_seeker=inferred_profile_for_supporter
            )
            second_cands = sup_api.generate_n(sup_prompt_2, n=LOOKAHEAD_K_SECOND, log_to_history=False)

            local_best = e1
            best_e2 = None
            e2_list = []

            for (st2, sup2, raw2, cot2) in (second_cands or []):
                seeker_hist_block_2 = history_for_seeker_block(
                    hist_inc_last
                    + [{"role": "supporter", "content": sup1, "strategy": st1}]
                    + [{"role": "seeker", "content": seeker_after_1_clean}]
                )
                seek_prompt_2 = build_seeker_prompt_sim(sim_background, seeker_hist_block_2, sup2)
                seeker_after_2_raw = sim_seek_api.generate_one(seek_prompt_2)

                end2 = has_end_any(seeker_after_2_raw)
                seeker_after_2_clean = strip_end_any(seeker_after_2_raw)
                e2 = 1.0 if end2 else val_api.score(seeker_after_2_clean, VALENCE_GENCFG)
                e2_list.append(e2)

                path_score = e1 + GAMMA * e2
                if path_score > local_best:
                    local_best = path_score
                    best_e2 = e2

        log_parsed = {"strategy": st1, "content": sup1}
        if isinstance(cot1, dict):
            log_parsed["cot"] = cot1
        logger.log_turn(
            {
                "ts": now_ts(),
                "who": "supporter(sim)",
                "round": round_id,
                "sim_layer": 1,
                "model": sup_api.model,
                "prompt": sup_prompt,
                "raw": raw1,
                "parsed": log_parsed,
                "valence_path_score": local_best,
                "valence_e1": e1,
                "valence_best_e2": best_e2,
                "valence_e2_list": e2_list,
            }
        )
        if SHOW_MODEL_OUTPUT:
            tqdm.write(f"[SIM][Supporter L1] {trunc(sup1)} (path_score={local_best:.2f}, e2_list={e2_list})")

        if local_best > best_score:
            best_score = local_best
            best_first = sup1
            best_first_st = st1

    if SHOW_MODEL_OUTPUT and best_first is not None:
        tqdm.write(f"[Round {round_id}] Lookahead pick -> {trunc(best_first)} (score={best_score:.2f})")
    return {"role": "supporter", "content": best_first, "strategy": best_first_st}


# ================== Real Seeker call ==================
def ask_real_seeker_5mini(client_5mini: OpenAI, prompt: str) -> str:
    extra = {}
    if REASONING_EFFORT_5MINI:
        extra["reasoning_effort"] = REASONING_EFFORT_5MINI
    if VERBOSITY_5MINI:
        extra["verbosity"] = VERBOSITY_5MINI
    resp = client_5mini.chat.completions.create(
        model=MODEL_5mini,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    msg = resp.choices[0].message
    txt = (msg.content or "").strip()
    if not txt:
        raise RuntimeError("Real Seeker (5mini) returned empty content.")
    return txt


# ================== Episode runner ==================
def run_episode(persona: Dict[str, Any], episode_dir: str, rounds_bar_desc: str) -> Dict[str, Any]:
    logger = BaseLogger(episode_dir)
    sup_api = SupporterAPI(BASE_URL_SUPPORTER, API_KEY_SUPPORTER, MODEL_SUPPORTER, logger)
    val_api = ValenceAPI(VALENCE_BASE, logger)

    summ_p = os.path.join(episode_dir, "summary.json")
    if os.path.exists(summ_p):
        final_hist = safe_load_json(os.path.join(episode_dir, "final_history.json"), default={"history": []})
        ended_by = safe_load_json(summ_p, default={}).get("ended_by", "max_rounds")
        return {"history": final_hist.get("history", []), "ended_by": ended_by}

    code, msg = val_api.health()
    logger.log_turn({"ts": now_ts(), "who": "valence_health", "status": code, "msg": msg})
    if PRINT_CALLS:
        tqdm.write(f"[Health] Valence({code}) {trunc(str(msg), 256)}")

    ckpt = safe_load_json(logger.ckpt_file, default=None)
    if ckpt and isinstance(ckpt, dict) and isinstance(ckpt.get("history"), list):
        history = ckpt["history"]
        inferred_profile_for_supporter = ckpt.get("inferred_profile_for_supporter", None)
        last_persona_round = ckpt.get("last_persona_round", None)
        if SHOW_MODEL_OUTPUT:
            tqdm.write(
                f"[Resume@{os.path.basename(episode_dir)}] history_steps={len(history)}  pairs={count_pairs(history)}"
            )
    else:
        history: List[Dict[str, Any]] = []
        inferred_profile_for_supporter = None
        last_persona_round = None
        logger.save_ckpt(
            {
                "ts": now_ts(),
                "persona": persona,
                "history": history,
                "max_rounds": MAX_ROUNDS,
                "inferred_profile_for_supporter": inferred_profile_for_supporter,
                "last_persona_round": last_persona_round,
            }
        )

    client_persona = OpenAI(
        base_url=BASE_URL_PERSONA,
        api_key=API_KEY_PERSONA,
        http_client=httpx.Client(base_url=BASE_URL_PERSONA, follow_redirects=True, timeout=REQ_TIMEOUT),
    )
    sim_seek_api = SeekerAPI(BASE_URL_SEEKER, API_KEY_SEEKER, MODEL_SEEKER)

    client_5mini = OpenAI(
        base_url=BASE_URL_5mini,
        api_key=API_KEY_5mini,
        http_client=httpx.Client(base_url=BASE_URL_5mini, follow_redirects=True, timeout=REQ_TIMEOUT),
    )

    if not history:
        print(f"\n[Ep] Supporter -> Seeker (Question): {OPENING_SUPPORTER} (strategy=Question)")
        history.append({"role": "supporter", "content": OPENING_SUPPORTER, "strategy": "Question"})
        logger.log_turn(
            {
                "ts": now_ts(),
                "who": "supporter(real)",
                "round": 0,
                "model": MODEL_SUPPORTER,
                "prompt": "[opening_fixed]",
                "raw": OPENING_SUPPORTER,
                "parsed": {"strategy": "Question", "content": OPENING_SUPPORTER},
            }
        )
        logger.save_ckpt(
            {
                "ts": now_ts(),
                "persona": persona,
                "history": history,
                "max_rounds": MAX_ROUNDS,
                "inferred_profile_for_supporter": None,
                "last_persona_round": None,
            }
        )

    done_pairs = count_pairs(history)
    rounds_bar = tqdm(total=MAX_ROUNDS, desc=rounds_bar_desc, ascii=TQDM_ASCII, initial=done_pairs)

    try:
        while True:
            done_pairs = count_pairs(history)
            if done_pairs >= MAX_ROUNDS:
                atomic_write_json(os.path.join(episode_dir, "final_history.json"), {"history": history})
                atomic_write_json(
                    summ_p,
                    {
                        "episode_dir": episode_dir,
                        "persona": persona,
                        "rounds_total": MAX_ROUNDS,
                        "ended_by": "max_rounds",
                        "finished_ts": now_ts(),
                    },
                )
                try:
                    os.remove(logger.ckpt_file)
                except Exception:
                    pass
                return {"history": history, "ended_by": "max_rounds"}

            last_role = history[-1]["role"]

            if last_role == "supporter":
                supporter_text = history[-1]["content"]
                dialogue_block = history_for_seeker_block(history)

                seeker_prompt = build_seeker_prompt_sim(
                    background=persona, dialogue_history_block=dialogue_block, supporter_query_text=supporter_text
                )

                r_next = done_pairs
                seeker_msg = ask_real_seeker_5mini(client_5mini, seeker_prompt)

                end_now = has_end_token(seeker_msg)
                seeker_msg_clean = strip_end_token(seeker_msg)
                print(f"[Round {r_next + 1}] Seeker: {seeker_msg_clean}" + (" [END]" if end_now else ""))

                v_sc = val_api.score(seeker_msg_clean, VALENCE_GENCFG)
                history.append({"role": "seeker", "content": seeker_msg_clean, "valence": v_sc})
                logger.log_turn(
                    {
                        "ts": now_ts(),
                        "who": "seeker(real_5mini)",
                        "round": r_next + 1,
                        "model": MODEL_5mini,
                        "prompt": seeker_prompt,
                        "raw": seeker_msg,
                        "clean": seeker_msg_clean,
                        "valence_score": v_sc,
                    }
                )
                rounds_bar.update(1)
                logger.save_ckpt(
                    {
                        "ts": now_ts(),
                        "persona": persona,
                        "history": history,
                        "max_rounds": MAX_ROUNDS,
                        "inferred_profile_for_supporter": locals().get("inferred_profile_for_supporter", None),
                        "last_persona_round": locals().get("last_persona_round", None),
                    }
                )

                if end_now:
                    rounds_bar.close()
                    atomic_write_json(os.path.join(episode_dir, "final_history.json"), {"history": history})
                    atomic_write_json(
                        summ_p,
                        {
                            "episode_dir": episode_dir,
                            "persona": persona,
                            "rounds_total": done_pairs + 1,
                            "ended_by": "seeker_end",
                            "finished_ts": now_ts(),
                        },
                    )
                    try:
                        os.remove(logger.ckpt_file)
                    except Exception:
                        pass
                    return {"history": history, "ended_by": "seeker_end"}

            else:
                next_pair = done_pairs + 1
                seeker_last = history[-1]["content"]

                # Persona inference
                if next_pair >= 3 and (last_persona_round is None or (next_pair - last_persona_round) >= 2):
                    try:
                        dj = dialog_json_for_persona_infer(history)
                        raw_persona_json_text = ask_persona_infer_local(client_persona, dj)
                        parsed = _try_parse_obj(raw_persona_json_text)
                        if not isinstance(parsed, dict):
                            raise RuntimeError(f"Persona Infer returned unparsable content: {trunc(raw_persona_json_text, 200)}")
                        cleaned = _clean_persona_json_keep_values(parsed)
                        inferred_profile_for_supporter = cleaned
                        last_persona_round = next_pair
                        logger.log_turn(
                            {
                                "ts": now_ts(),
                                "who": "persona_infer",
                                "round": next_pair,
                                "model": MODEL_PERSONA,
                                "raw_text": raw_persona_json_text,
                                "parsed_cleaned": cleaned,
                            }
                        )
                        if PRINT_CALLS:
                            tqdm.write(f"[PersonaInfer] acquired at round {next_pair}")
                        logger.save_ckpt(
                            {
                                "ts": now_ts(),
                                "persona": persona,
                                "history": history,
                                "max_rounds": MAX_ROUNDS,
                                "inferred_profile_for_supporter": inferred_profile_for_supporter,
                                "last_persona_round": last_persona_round,
                            }
                        )
                    except Exception as e:
                        logger.log_error(
                            {"ts": now_ts(), "where": "persona_infer", "round": next_pair, "error": repr(e)}
                        )

                # Lookahead
                if next_pair >= 3 and inferred_profile_for_supporter:
                    sim_background = build_sim_background_from_infer(persona, inferred_profile_for_supporter)
                    chosen_turn = choose_supporter_by_lookahead(
                        round_id=next_pair,
                        history_excl_last=[h for h in history[:-1]],
                        seeker_last=seeker_last,
                        sim_background=sim_background,
                        sup_api=sup_api,
                        sim_seek_api=sim_seek_api,
                        val_api=val_api,
                        inferred_profile_for_supporter=inferred_profile_for_supporter,
                        logger=logger,
                    )
                else:
                    sup_hist_list = history_for_supporter_list(history[:-1])
                    sup_prompt = build_supporter_prompt(sup_hist_list, "seeker", seeker_last, profile_seeker=inferred_profile_for_supporter)
                    st, sup_text, raw, cot = sup_api.generate_one(sup_prompt)
                    chosen_turn = {"role": "supporter", "content": sup_text, "strategy": st}
                    if isinstance(cot, dict):
                        chosen_turn["cot"] = cot

                history.append(chosen_turn)
                if SHOW_MODEL_OUTPUT:
                    tqdm.write(f"[Round {done_pairs + 1}] Supporter -> Seeker: {trunc(chosen_turn['content'])} (strategy={chosen_turn.get('strategy', 'Others')})")

                logger.save_ckpt(
                    {
                        "ts": now_ts(),
                        "persona": persona,
                        "history": history,
                        "max_rounds": MAX_ROUNDS,
                        "inferred_profile_for_supporter": inferred_profile_for_supporter,
                        "last_persona_round": last_persona_round,
                    }
                )

    except Exception as e:
        with open(os.path.join(episode_dir, "fatal_error.txt"), "w", encoding="utf-8") as f:
            f.write(now_ts() + "\n" + repr(e) + "\n" + traceback.format_exc())
        logger.log_error({"ts": now_ts(), "where": "run_episode", "error": repr(e), "trace": traceback.format_exc()})
        raise
    finally:
        try:
            rounds_bar.close()
        except Exception:
            pass


# ================== Preflight ==================
def preflight_check() -> None:
    print("\n========== Preflight: API health checks ==========")

    # 1) Supporter
    try:
        cli_sup = OpenAI(
            base_url=BASE_URL_SUPPORTER,
            api_key=API_KEY_SUPPORTER,
            http_client=httpx.Client(base_url=BASE_URL_SUPPORTER, follow_redirects=True, timeout=REQ_TIMEOUT),
        )
        resp = cli_sup.chat.completions.create(
            model=MODEL_SUPPORTER,
            temperature=0.0,
            messages=[{"role": "user", "content": "Return a brief supportive sentence."}],
            max_tokens=64,
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            raise RuntimeError("Supporter returned empty content.")
        print(f"[OK] Supporter API ({MODEL_SUPPORTER}) -> {trunc(out, 60)}")
    except Exception as e:
        print(f"[FAIL] Supporter API ({MODEL_SUPPORTER}) check failed: {e}")
        sys.exit(12)

    # 2) Real Seeker (5mini)
    try:
        extra = {}
        if REASONING_EFFORT_5MINI:
            extra["reasoning_effort"] = REASONING_EFFORT_5MINI
        if VERBOSITY_5MINI:
            extra["verbosity"] = VERBOSITY_5MINI
        cli_5mini = OpenAI(
            base_url=BASE_URL_5mini,
            api_key=API_KEY_5mini,
            http_client=httpx.Client(base_url=BASE_URL_5mini, follow_redirects=True, timeout=REQ_TIMEOUT),
        )
        resp = cli_5mini.chat.completions.create(
            model=MODEL_5mini, temperature=0.0, messages=[{"role": "user", "content": "Ping."}], **extra
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            raise RuntimeError("Real Seeker 5mini returned empty.")
        print(f"[OK] Real Seeker API ({MODEL_5mini}) -> {trunc(out, 60)}")
    except Exception as e:
        print(f"[FAIL] Real Seeker API ({MODEL_5mini}) check failed: {e}")
        sys.exit(13)

    # 3) Persona Infer
    try:
        cli_persona = OpenAI(
            base_url=BASE_URL_PERSONA,
            api_key=API_KEY_PERSONA,
            http_client=httpx.Client(base_url=BASE_URL_PERSONA, follow_redirects=True, timeout=REQ_TIMEOUT),
        )
        out = ask_persona_infer_local(cli_persona, json.dumps([{"role": "seeker", "content": "Hi"}]))
        if not out:
            raise RuntimeError("Persona Infer returned empty.")
        print(f"[OK] Persona API ({MODEL_PERSONA}) -> {trunc(out, 60)}")
    except Exception as e:
        print(f"[FAIL] Persona API ({MODEL_PERSONA}) check failed: {e}")
        sys.exit(14)

    # 4) Sim Seeker
    try:
        cli_lseek = OpenAI(
            base_url=BASE_URL_SEEKER,
            api_key=API_KEY_SEEKER,
            http_client=httpx.Client(base_url=BASE_URL_SEEKER, follow_redirects=True, timeout=REQ_TIMEOUT),
        )
        resp = cli_lseek.chat.completions.create(
            model=MODEL_SEEKER, temperature=0.0, messages=[{"role": "user", "content": "Ping"}], max_tokens=64
        )
        out = (resp.choices[0].message.content or "").strip()
        if not out:
            raise RuntimeError("Local Sim Seeker returned empty.")
        print(f"[OK] Sim Seeker API ({MODEL_SEEKER}) -> {trunc(out, 60)}")
    except Exception as e:
        print(f"[FAIL] Sim Seeker API ({MODEL_SEEKER}) check failed: {e}")
        sys.exit(15)

    # 5) Valence
    try:
        cli = httpx.Client(timeout=REQ_TIMEOUT, follow_redirects=True)
        r = cli.get(f"{VALENCE_BASE}/health")
        r.raise_for_status()
        print(f"[OK] Valence /health -> {trunc(str(r.json()), 100)}")
    except Exception as e:
        print(f"[FAIL] Valence API check failed: {e}")
        sys.exit(16)

    print("========== Preflight passed ==========\n")


# ================== Main ==================
def main():
    preflight_check()
    run_dir = select_run_dir()
    ensure_dir(run_dir)

    meta_p = os.path.join(run_dir, "run_meta.json")
    meta = safe_load_json(meta_p, default=None)
    if not meta:
        meta = {
            "run_id": os.path.basename(run_dir).replace("run_", "", 1),
            "created": now_ts(),
            "config": {
                "MAX_ROUNDS": MAX_ROUNDS,
                "BASE_URL_SUPPORTER": BASE_URL_SUPPORTER,
                "MODEL_SUPPORTER": MODEL_SUPPORTER,
                "MODEL_5mini_REAL": MODEL_5mini,
                "MODEL_SEEKER_SIM": MODEL_SEEKER,
                "PERSONA_JSON": PERSONA_JSON,
            },
            "status": "running",
        }
        atomic_write_json(meta_p, meta)
    else:
        meta["last_resume"] = now_ts()
        atomic_write_json(meta_p, meta)

    personas_obj = load_json(PERSONA_JSON)
    personas: List[Any] = list(p for _, p in iter_personas(personas_obj))

    n = len(personas)
    if PERSONA_SEGMENT != "all" and n >= 3:
        third = n // 3
        if PERSONA_SEGMENT == "first":
            personas = personas[:third]
        elif PERSONA_SEGMENT == "middle":
            personas = personas[third : 2 * third]
        elif PERSONA_SEGMENT == "last":
            personas = personas[2 * third :]
        print(f"[Persona segment] {PERSONA_SEGMENT} -> count={len(personas)}")
    else:
        print(f"[Persona segment] all -> count={n}")

    all_convs: List[Dict[str, Any]] = load_all_convs(run_dir)
    done_set = finished_persona_indices(run_dir, personas)

    for pid, persona in tqdm(list(enumerate(personas)), desc="Episodes", ascii=TQDM_ASCII):
        ep_dir = ensure_dir(os.path.join(run_dir, f"episode_{pid:03d}"))
        atomic_write_json(os.path.join(ep_dir, "background.json"), persona)

        if pid in done_set:
            tqdm.write(f"[Skip] persona_index={pid} already finished.")
            continue

        try:
            result = run_episode(persona, ep_dir, rounds_bar_desc=f"Rounds@Ep{pid}")
            hist = result["history"]
            rounds = count_pairs(hist)
            end_flag = (result.get("ended_by") == "seeker_end")
            entry = {
                "persona_index": pid,
                "conv_id": uuid.uuid4().hex[:8],
                "ended_by": ("seeker_end" if end_flag else "max_rounds"),
                "rounds": rounds,
                "seeker_profile": persona,
                "dialog": hist,
            }
            all_convs = upsert_conv(all_convs, pid, entry)
            save_all_convs(run_dir, all_convs)
        except Exception as e:
            print(f"\n[Persona {pid}] ERROR:", e)
            traceback.print_exc()
            entry = {"persona_index": pid, "conv_id": uuid.uuid4().hex[:8], "error": str(e), "traceback": traceback.format_exc()}
            all_convs = upsert_conv(all_convs, pid, entry)
            save_all_convs(run_dir, all_convs)

    per_round_scores: Dict[int, List[float]] = {}
    all_scores: List[float] = []
    ended_by_end = 0
    for c in all_convs:
        hist = c.get("dialog", []) or []
        if c.get("ended_by") == "seeker_end":
            ended_by_end += 1
        r = 0
        for i in range(1, len(hist)):
            if hist[i - 1].get("role") == "supporter" and hist[i].get("role") == "seeker":
                r += 1
                sc = hist[i].get("valence", None)
                if isinstance(sc, (int, float)):
                    per_round_scores.setdefault(r, []).append(float(sc))
                    all_scores.append(float(sc))

    avg_curve = {r: (sum(v) / len(v) if v else None) for r, v in sorted(per_round_scores.items())}
    overall_stats = {"count": len(all_scores), "mean": (sum(all_scores) / len(all_scores) if all_scores else None)}
    stats = {
        "run_dir": run_dir,
        "total_dialogues": len(all_convs),
        "ended_by_END": ended_by_end,
        "per_round_avg_valence": avg_curve,
        "overall_valence_stats": overall_stats,
    }
    atomic_write_json(os.path.join(run_dir, "global_stats.json"), stats)
    if isinstance(meta, dict):
        meta["status"] = "finished"
        meta["finished_ts"] = now_ts()
        atomic_write_json(meta_p, meta)

    print("\n========== Global Stats ==========")
    print(f"Total dialogues: {len(all_convs)}")
    print(f"Ended by [END]: {ended_by_end}")
    print("Mean Valence:", overall_stats["mean"])
    print(f"Saved -> {run_dir}")


if __name__ == "__main__":
    main()
