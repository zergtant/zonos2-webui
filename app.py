#!/usr/bin/env python3
"""Zonos2 WebUI — Voice Management + Voice Cloning

Two-tab Gradio app:
  - Tab 1 语音克隆: select role + embedding, enter text, generate TTS
  - Tab 2 音色管理: create roles, upload audio, extract embeddings, manage voices

Usage:  conda activate zonos2 && python app.py
"""

import json, random, shutil, time, threading
from pathlib import Path

# ── Env (before any heavy imports) ──
import os
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"
VOICES_DIR = PROJECT_DIR / "voices"

os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_SSR_MODE", "false")
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0,::1"

for d in ("dac_cache", "huggingface", "matplotlib"):
    (CACHE_DIR / d).mkdir(parents=True, exist_ok=True)

import gradio as gr
import librosa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn.functional as F, torchaudio
from transformers import AutoModel
from zonos2.tokenizer.textnorm import TTSTextNormalizer, SERVER_TO_NEMO_LANG
from zonos2.message.tts import TTSSamplingParams, TTSUserMsg
from zonos2.tts.llm import TTSLLM

# ── Constants ──
MODEL_PATH = str(CACHE_DIR / "models")
QWEN3_MODEL_PATH = str(CACHE_DIR / "Qwen3-Voice-Embedding-12Hz-1.7B")
SAMPLE_RATE = 44100; FRAMES_PER_SECOND = SAMPLE_RATE / 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max
MIN_AUDIO_SEC = 1.0; MAX_AUDIO_SEC = 30.0

LANGUAGES = [("English (US)","en_us"),("English (UK)","en_gb"),("French","fr_fr"),
             ("German","de"),("Spanish","es"),("Italian","it"),("Portuguese (BR)","pt_br"),
             ("Japanese","ja"),("Mandarin","cmn"),("Korean","ko")]
RATE_CHOICES = ["Auto","0-8","8-11","11-14","14-17","17-21","21-28","28-40","40+"]

_NORMALIZER = TTSTextNormalizer()
threading.Thread(target=_NORMALIZER.warmup, daemon=True).start()

# ═══════════════════════════ ZonosTTSLLM ═══════════════════════════
class ZonosTTSLLM(TTSLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.speaker_embedding: torch.Tensor | None = None
        self.clean_speaker_background = False
        self.accurate_mode = True
    def offline_receive_msg(self, blocking=False):
        msgs = super().offline_receive_msg(blocking)
        for msg in msgs:
            if isinstance(msg, TTSUserMsg):
                msg.speaker_embedding = self.speaker_embedding
                msg.clean_speaker_background = self.clean_speaker_background
                msg.accurate_mode = self.accurate_mode
        return msgs

# ═══════════════════════════ Singletons ═══════════════════════════
_TTS = None; _TTS_LOCK = threading.Lock(); _EMBEDDER = None

def _get_tts():
    global _TTS
    if _TTS is not None: return _TTS
    with _TTS_LOCK:
        if _TTS is not None: return _TTS
        print(f"Loading TTS model from {MODEL_PATH} …", flush=True)
        t0 = time.perf_counter()
        _TTS = ZonosTTSLLM(model_path=MODEL_PATH, decode_audio=True,
                           cuda_graph_max_bs=0, max_running_req=4,
                           max_extend_tokens=4096, memory_ratio=0.75, use_pynccl=False)
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    return _TTS

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None: return _EMBEDDER
    print(f"Loading Qwen3 from {QWEN3_MODEL_PATH} …", flush=True)
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(QWEN3_MODEL_PATH, trust_remote_code=True)
    model.to(DEVICE); model.eval()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256,
        f_min=0.0, f_max=12000.0, n_mels=128, power=1.0, center=False,
        norm="slaney", mel_scale="slaney").to(DEVICE)
    _EMBEDDER = {"model": model, "mel": mel}
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    return _EMBEDDER

# ═══════════════════════════ Voice Helpers ═══════════════════════════

def list_roles():
    if not VOICES_DIR.exists(): return []
    return sorted(d.name for d in VOICES_DIR.iterdir() if d.is_dir() and not d.name.startswith("."))

def _rp(role): return VOICES_DIR / role
def _ed(role):
    p = _rp(role)/"embeddings"; p.mkdir(parents=True, exist_ok=True); return p
def _wd(role):
    p = _rp(role)/"wav"; p.mkdir(parents=True, exist_ok=True); return p
def _mp(role): return _rp(role)/"metadata.json"

def _ensure_role(role):
    _rp(role).mkdir(parents=True, exist_ok=True); _ed(role); _wd(role)

def _load_meta(role):
    path = _mp(role)
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError): pass
    return {"files":[],"mean_stats":None,"cosine_similarity":None,"similarity_labels":[]}

def _save_meta(role, meta):
    _mp(role).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

def _next_id(role):
    ed = _ed(role); max_n = 0
    for f in ed.glob("*.npy"):
        if f.stem == "mean": continue
        try: n = int(f.stem); max_n = max(max_n, n)
        except ValueError: continue
    return f"{max_n+1:03d}"

# ═══════════════════════════ CRUD ═══════════════════════════

def delete_role(role):
    rp = _rp(role)
    if not rp.exists(): return f"⚠️ 角色「{role}」不存在"
    shutil.rmtree(rp)
    return f"✅ 已删除角色「{role}」"

def extract_embedding(audio_path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    dur = len(y)/sr
    if dur < MIN_AUDIO_SEC: raise ValueError(f"音频太短: {dur:.1f}s（最少{MIN_AUDIO_SEC:.0f}s）")
    if dur > MAX_AUDIO_SEC: raise ValueError(f"音频太长: {dur:.0f}s（最长{MAX_AUDIO_SEC:.0f}s）")
    wav = torch.from_numpy(y).unsqueeze(0).to(DEVICE, torch.float32)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000).to(DEVICE)(wav)
    emb = _get_embedder()
    pad = (1024-256)//2
    wav_p = F.pad(wav.unsqueeze(1), (pad,pad), mode="reflect").squeeze(1)
    mel = emb["mel"](wav_p); mel = torch.log(torch.clamp(mel, min=1e-5)); mel = mel.transpose(1,2)
    with torch.inference_mode():
        out = emb["model"](input_values=mel).last_hidden_state
    vec = out.squeeze(0).float().cpu()
    if vec.ndim == 2: vec = vec.mean(0)
    if vec.numel() != 2048: raise ValueError(f"Embedding 维度异常: {vec.shape}")
    return vec.numpy().reshape(2048).astype(np.float32), dur

def _cosine_matrix(embs, labels):
    n = len(embs); mat = np.zeros((n,n), dtype=np.float32)
    for i in range(n):
        for j in range(i,n):
            a,b = embs[i], embs[j]
            mat[i,j] = mat[j,i] = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    return mat, labels

def _rebuild_meta(role):
    ed = _ed(role); meta = _load_meta(role)
    # Build set of known file IDs from metadata
    known = {f["id"]: f for f in meta.get("files",[])}
    # Scan disk for .npy files
    npy_files = sorted(f for f in ed.glob("*.npy") if f.stem != "mean")
    new_files = []
    for nf in npy_files:
        fid = nf.stem
        if fid in known:
            new_files.append(known[fid])
        else:
            # New file found on disk but not in metadata → create minimal entry
            emb = np.load(str(nf))
            new_files.append({
                "id": fid,
                "wav": f"wav/{fid}.wav",
                "npy": f"embeddings/{fid}.npy",
                "original": "",
                "duration_s": 0,
                "mean": float(emb.mean()),
                "std": float(emb.std()),
            })
    meta["files"] = new_files

    # Collect embeddings and labels (for mean + similarity)
    all_embs, all_labels = [], []
    for f_entry in new_files:
        npy_path = _rp(role)/f_entry["npy"]
        if npy_path.exists():
            all_embs.append(np.load(str(npy_path)))
            all_labels.append(f_entry["id"])
    mean_path = ed/"mean.npy"
    if len(all_embs) >= 2:
        mean_emb = np.mean(all_embs, axis=0).astype(np.float32)
        np.save(str(mean_path), mean_emb)
        all_embs.append(mean_emb); all_labels.append("mean")
        meta["mean_stats"] = {"mean": float(mean_emb.mean()), "std": float(mean_emb.std())}
    elif len(all_embs) == 1:
        if mean_path.exists(): mean_path.unlink()
        meta["mean_stats"] = {"mean": float(all_embs[0].mean()), "std": float(all_embs[0].std())}
    else:
        if mean_path.exists(): mean_path.unlink()
        meta["mean_stats"] = None
    if len(all_embs) >= 2:
        mat, labels = _cosine_matrix(all_embs, all_labels)
        meta["cosine_similarity"] = mat.tolist(); meta["similarity_labels"] = labels
    else:
        meta["cosine_similarity"] = None; meta["similarity_labels"] = []
    _save_meta(role, meta); return meta

def import_audio(role_name, audio_file):
    role = (role_name or "").strip()
    if not role: raise gr.Error("请选择或输入角色名")
    _ensure_role(role)
    if audio_file is None: raise gr.Error("请上传一个 .wav 文件")
    fp = Path(audio_file)
    if fp.suffix.lower() != ".wav": raise gr.Error("仅支持 .wav 格式")
    orig = fp.name
    try: embedding, dur = extract_embedding(str(fp))
    except ValueError as e: raise gr.Error(str(e))
    fid = _next_id(role)
    np.save(str(_ed(role)/f"{fid}.npy"), embedding)
    shutil.copy2(str(fp), str(_wd(role)/f"{fid}.wav"))
    meta = _load_meta(role)
    meta.setdefault("files",[]).append({"id":fid, "wav":f"wav/{fid}.wav", "npy":f"embeddings/{fid}.npy",
        "original":orig, "duration_s":round(dur,2),
        "mean":float(embedding.mean()), "std":float(embedding.std())})
    _save_meta(role, meta)
    meta = _rebuild_meta(role)
    return role, json.dumps(meta, ensure_ascii=False), f"✅ 已保存为 {fid}.npy" + (" · mean.npy 已更新" if len(meta.get("files",[]))>=2 else "")

def delete_audio_files(role, fids):
    if isinstance(fids, str): fids = [fids]
    for fid in fids:
        for ext in (".npy",".wav"):
            p = _rp(role)/("embeddings" if ext==".npy" else "wav")/f"{fid}{ext}"
            if p.exists(): p.unlink()
    meta = _rebuild_meta(role)
    return json.dumps(meta, ensure_ascii=False), f"✅ 已删除 {fids[0]}" if len(fids)==1 else f"✅ 已删除 {len(fids)} 个"

# ═══════════════════════════ Plot ═══════════════════════════

def plot_heatmap(meta_json):
    if not meta_json: return None
    try: meta = json.loads(meta_json)
    except: return None
    sim = meta.get("cosine_similarity"); labels = meta.get("similarity_labels",[])
    if not sim or len(labels)<2: return None
    arr = np.array(sim, dtype=np.float32); n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4,n*1.5), max(3.5,n*1.3)))
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine")
    short = [l[:8] for l in labels]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(n):
        for j in range(n):
            c = "white" if abs(arr[i,j])>0.6 else "black"
            ax.text(j,i,f"{arr[i,j]:.3f}", ha="center", va="center", fontsize=6, color=c)
    fig.tight_layout(); return fig

# ═══════════════════════════ Role detail ═══════════════════════════

def get_role_detail(role):
    if not role: return "{}", [], None
    meta = _load_meta(role); mj = json.dumps(meta, ensure_ascii=False)
    n = len(meta.get("files",[]))
    rows = [[f["id"], f.get("original",""), f"{f['duration_s']}s",
             f"{f['mean']:.4f}", f"{f['std']:.4f}"] for f in meta.get("files",[])]
    hm = plot_heatmap(mj) if n>=2 else None
    return mj, rows, hm

# ═══════════════════════════ Clone helpers ═══════════════════════════

def normalize_text(text, lang):
    text = (text or "").strip()
    if not text: raise gr.Error("请输入文本")
    if len(text)>5000: raise gr.Error("文本过长，控制在5000字符以内")
    if not lang or lang not in SERVER_TO_NEMO_LANG: return text
    return _NORMALIZER.normalize(text, lang)

def _emb_choices(role):
    meta = _load_meta(role); files = meta.get("files",[])
    choices, default = [], None
    for f in files:
        choices.append((f"{f['id']} ({f['duration_s']}s | μ={f['mean']:.4f})", f["id"]))
    if len(files)>=2:
        ms = meta.get("mean_stats",{})
        choices.append((f"mean (×{len(files)} | μ={ms.get('mean',0):.4f})", "mean"))
        default = "mean"
    elif len(files)==1: default = files[0]["id"]
    return choices, default

# ═══════════════════════════ Generate ═══════════════════════════

def generate(text, lang, role, emb_id, rate,
             accurate, clean_bg, max_sec, temp, topk, minp, rep_pen, seed,
             progress=gr.Progress()):
    try:
        text = normalize_text(text, lang)
        seed = int(seed)
        if seed == 0: seed = random.randint(1, MAX_SEED)

        embedding = None; src = "TTS"
        if role and emb_id:
            ed = _ed(role)
            ep = ed/"mean.npy" if emb_id=="mean" else ed/f"{emb_id}.npy"
            if not ep.exists(): return gr.update(), f"❌ 找不到 {ep.name}"
            src = f"{role}/{emb_id}"
            progress(0.05, desc="加载 speaker embedding …")
            emb_data = np.load(str(ep))
            if emb_data.shape != (2048,): return gr.update(), f"❌ Embedding 维度异常: {emb_data.shape}"
            embedding = torch.from_numpy(emb_data).to(torch.float32)

        progress(0.1, desc="初始化 TTS 模型 …")
        tts = _get_tts(); torch.cuda.set_stream(tts.stream)
        tts.speaker_embedding = embedding
        tts.clean_speaker_background = bool(clean_bg) and embedding is not None
        tts.accurate_mode = bool(accurate)

        params = TTSSamplingParams(
            temperature=float(temp), topk=int(topk), top_p=0.0, min_p=float(minp),
            max_tokens=int(float(max_sec)*FRAMES_PER_SECOND),
            repetition_window=50, repetition_penalty=float(rep_pen),
            repetition_codebooks=8, seed=None if seed<0 else seed)
        rate_idx = None if rate=="Auto" else RATE_CHOICES.index(rate)-1

        progress(0.3, desc="生成语音 …")
        result = tts.generate_one(text, params, decode_audio=True,
                                   speaking_rate_bucket=rate_idx, quality_buckets=None)

        ab = result.get("audio")
        if not ab or len(ab)==0: return gr.update(), "❌ 模型没有生成音频，尝试不同的随机种子或较短的文本"
        audio = np.clip(np.nan_to_num(np.frombuffer(ab, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        sr = result.get("sample_rate", SAMPLE_RATE); dur = len(audio)/sr
        status = f"✅ 已生成 ({dur:.1f}s, seed={seed})" + (f" · 音色: {src}" if embedding is not None else " · 纯 TTS")
        return (sr, audio), status
    except Exception as e:
        return gr.update(), f"❌ 生成失败: {e}"

# ════════════════════════════════ UI ════════════════════════════════

_EMPTY_PLOT = None

def _empty_plot():
    """Return a cached blank figure so gr.Plot never gets None."""
    global _EMPTY_PLOT
    if _EMPTY_PLOT is None:
        fig, ax = plt.subplots(figsize=(1,1))
        ax.axis("off"); fig.tight_layout(pad=0); _EMPTY_PLOT = fig
    return _EMPTY_PLOT

with gr.Blocks(title="Zonos2 WebUI") as demo:

    # ── States ──
    cur_role_st = gr.State("")
    cur_meta_st = gr.State("{}")
    sel_file_st = gr.State("")

    gr.Markdown("# 🗣️ Zonos2 WebUI — 语音克隆 & 音色管理")

    # ════════════════ Tab 1: 语音克隆 ════════════════
    with gr.Tab("🗣️ 语音克隆"):
        with gr.Row():
            # ── Left: role + embedding + preview ──
            with gr.Column(scale=1):
                gr.Markdown("### 选择角色")
                clone_refresh_btn = gr.Button("🔄 刷新角色列表")
                clone_role_dd = gr.Dropdown(label="角色", choices=[])
                clone_emb_radio = gr.Radio(label="选用 embedding", choices=[])
                clone_preview = gr.Audio(label="试听", type="filepath", interactive=False, visible=True)

            # ── Right: text + params + generate ──
            with gr.Column(scale=2):
                clone_text = gr.Textbox(label="输入文本", lines=4,
                    value="The first explorers landed just after sunrise, carrying maps, coffee, and impossible optimism.")
                clone_rate = gr.Dropdown(choices=RATE_CHOICES, value="Auto", label="语速")

                gr.Markdown("**参数**")
                with gr.Row():
                    acc_mode = gr.Checkbox(value=True, label="Accurate mode")
                    clean_bg = gr.Checkbox(value=True, label="Clean background")
                    clone_lang = gr.Dropdown(
                        choices=[("不规范化","")] + LANGUAGES, value="",
                        label="文本规范化语言")
                with gr.Row():
                    max_sec = gr.Slider(2, 60, 30, step=1, label="最大时长（秒）")
                    seed_n = gr.Number(value=0, precision=0, label="Seed（0=随机）")
                    temp = gr.Slider(0.1, 2.0, 1.15, step=0.05, label="Temperature")
                with gr.Row():
                    topk = gr.Slider(1, 512, 106, step=1, label="Top-k")
                    minp = gr.Slider(0.0, 0.5, 0.18, step=0.01, label="Min-p")
                    rep_pen = gr.Slider(1.0, 2.0, 1.2, step=0.05, label="Repetition penalty")

                gen_btn = gr.Button("🎯 生成语音", variant="primary")

        gr.Markdown("### 🎵 生成结果")
        audio_out = gr.Audio(label="生成语音", type="numpy")
        status_out = gr.Textbox(label="状态", interactive=False)

    # ════════════════ Tab 2: 音色管理 ════════════════
    with gr.Tab("🎤 音色管理"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 我的音色库")
                role_dd = gr.Dropdown(label="选择角色", choices=[], interactive=True)
                refresh_roles_btn = gr.Button("🔄 刷新角色列表")
                del_role_btn = gr.Button("🗑 删除选中角色", variant="stop")

                gr.Markdown("### 📥 导入音频")
                import_role_dd = gr.Dropdown(
                    label="选择角色（可输入新名称创建）", choices=[], allow_custom_value=True)
                audio_file_inp = gr.File(label="上传 .wav 文件（1~30秒）", file_types=[".wav"])
                import_btn = gr.Button("📥 导入并提取", variant="primary")

            with gr.Column(scale=2):
                detail_md = gr.Markdown("### 请选择一个角色")
                file_tbl = gr.Dataframe(
                    headers=["ID","原始文件名","时长","μ","σ"],
                    label="音频文件列表（点击行选择）", interactive=False, type="array")
                preview_audio = gr.Audio(label="试听选中（点击上方列表行播放）", type="filepath", interactive=False, visible=True)
                del_file_btn = gr.Button("🗑 删除选中", variant="stop")
                sim_plot = gr.Plot(label="余弦相似度热力图", value=_empty_plot())

        import_st = gr.Textbox(label="状态", interactive=False)

    # ════════════════ Events — Tab 2: 音色管理 ════════════════

    def _on_refresh_roles():
        """Refresh role dropdowns in 音色管理 tab."""
        roles = list_roles()
        c = roles if roles else []
        return gr.update(choices=c, value=None), gr.update(choices=c)

    refresh_roles_btn.click(fn=_on_refresh_roles, inputs=[],
                            outputs=[role_dd, import_role_dd])

    def _on_role_dd_change(role):
        """User selects a role from the dropdown → show detail."""
        if not role:
            return ("{}", gr.update(value=[]), _empty_plot(),
                    "### 请选择一个角色", "", gr.update(value=None))
        mj, rows, hm = get_role_detail(role)
        return (mj, gr.update(value=rows), (hm or _empty_plot()),
                f"### 角色: {role}", role, gr.update(value=None))

    role_dd.change(fn=_on_role_dd_change, inputs=[role_dd],
                   outputs=[cur_meta_st, file_tbl, sim_plot, detail_md, cur_role_st, preview_audio])

    def _on_del_role(role):
        if not role:
            return (gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), "❌ 请先选择一个角色")
        msg = delete_role(role)
        roles = list_roles(); c = roles if roles else []
        return (gr.update(choices=c, value=None), gr.update(choices=c),
                gr.update(choices=c),
                "{}", gr.update(value=[]), _empty_plot(),
                "### 请选择一个角色", "", msg)

    del_role_btn.click(fn=_on_del_role, inputs=[cur_role_st],
        outputs=[role_dd, import_role_dd, clone_role_dd,
                 cur_meta_st, file_tbl, sim_plot, detail_md, cur_role_st, import_st])

    def _on_file_tbl_click(evt: gr.SelectData, mj, role):
        if evt.index is None or not mj or not role: return "", gr.update(value=None)
        try: meta = json.loads(mj)
        except: return "", gr.update(value=None)
        files = meta.get("files",[])
        idx = evt.index[0] if isinstance(evt.index, (list,tuple)) else evt.index
        if idx is None or idx >= len(files): return "", gr.update(value=None)
        fid = files[idx]["id"]
        wp = _rp(role)/files[idx]["wav"]
        return fid, gr.update(value=str(wp)) if wp.exists() else gr.update(value=None)

    file_tbl.select(fn=_on_file_tbl_click, inputs=[cur_meta_st, cur_role_st],
                    outputs=[sel_file_st, preview_audio])

    def _on_del_file(fid, role):
        if not fid or not role:
            return (gr.update(), gr.update(value=[]), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), "❌ 请先在音频列表中点击选择要删除的音频", gr.update())
        new_mj, st = delete_audio_files(role, [fid])
        meta = json.loads(new_mj) if new_mj else {"files":[]}
        rows = [[f["id"], f.get("original",""), f"{f['duration_s']}s",
                 f"{f['mean']:.4f}", f"{f['std']:.4f}"] for f in meta.get("files",[])]
        hm = plot_heatmap(new_mj) if len(meta.get("files",[]))>=2 else None
        roles = list_roles(); c = roles if roles else []
        return (new_mj, gr.update(value=rows), hm or _empty_plot(),
                f"### 角色: {role}" if role else "### 请选择一个角色",
                gr.update(choices=c), gr.update(choices=c), gr.update(choices=c),
                "", st, gr.update(value=None))

    del_file_btn.click(fn=_on_del_file, inputs=[sel_file_st, cur_role_st],
        outputs=[cur_meta_st, file_tbl, sim_plot, detail_md,
                 role_dd, import_role_dd, clone_role_dd, sel_file_st, import_st, preview_audio])

    def _on_import(role, af):
        role = (role or "").strip()
        if not role:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    "❌ 角色名不能为空，请选择或输入角色名", gr.update())
        if af is None:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    "❌ 请上传一个 .wav 文件", gr.update())
        try:
            new_role, mj, st = import_audio(role, af)
        except Exception as e:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    f"❌ 导入失败: {e}", gr.update())
        _, rows, hm = get_role_detail(new_role)
        roles = list_roles(); c = roles if roles else []
        return (mj, gr.update(value=rows), hm or _empty_plot(),
                f"### 角色: {new_role}",
                gr.update(choices=c, value=new_role),
                gr.update(choices=c, value=None),
                gr.update(choices=c, value=new_role),
                new_role, st, None)

    import_btn.click(fn=_on_import, inputs=[import_role_dd, audio_file_inp],
        outputs=[cur_meta_st, file_tbl, sim_plot, detail_md,
                 role_dd, import_role_dd, clone_role_dd, cur_role_st, import_st, audio_file_inp])

    # ════════════════ Events — Tab 1: 语音克隆 ════════════════

    # Refresh button: populate role dropdown + radio (preview handled by radio.change)
    def _on_refresh():
        roles = list_roles()
        c = roles if roles else []
        first = c[0] if c else None
        if first:
            choices, default = _emb_choices(first)
            radio_upd = gr.update(choices=choices, value=default)
        else:
            radio_upd = gr.update(choices=[], value=None)
        return gr.update(choices=c, value=first), radio_upd

    clone_refresh_btn.click(fn=_on_refresh, inputs=[],
                            outputs=[clone_role_dd, clone_emb_radio])

    # Role changed → only update radio (preview handled by radio.change)
    def _on_clone_role(role):
        if not role:
            return gr.update(choices=[], value=None)
        choices, default = _emb_choices(role)
        return gr.update(choices=choices, value=default)

    clone_role_dd.change(fn=_on_clone_role, inputs=[clone_role_dd],
                         outputs=[clone_emb_radio])

    # Radio changed (programmatic or user click) → auto preview
    # mean has no wav → returns None
    def _on_radio_change(role, eid):
        if not role or not eid or eid == "mean":
            return gr.update(value=None)
        for f in _load_meta(role).get("files", []):
            if f["id"] == eid:
                wp = _rp(role) / f["wav"]
                if wp.exists():
                    return gr.update(value=str(wp))
        return gr.update(value=None)

    clone_emb_radio.change(fn=_on_radio_change,
                           inputs=[clone_role_dd, clone_emb_radio],
                           outputs=[clone_preview])

    gen_btn.click(fn=generate, inputs=[
        clone_text, clone_lang, clone_role_dd, clone_emb_radio, clone_rate,
        acc_mode, clean_bg, max_sec, temp, topk, minp, rep_pen, seed_n],
        outputs=[audio_out, status_out])

# ═══════════════════════════ Main ═══════════════════════════
if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=7866)
