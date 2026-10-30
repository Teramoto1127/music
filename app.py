"""MV / 音声を読み込んで「どんな曲か」を分析・解説するStreamlitアプリ(MVP)

起動: streamlit run app.py
"""
import os
import subprocess
import tempfile
from collections import Counter

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

SR = 22050
HOP = 512
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler のキープロファイル
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---------- 音声抽出 ----------
def extract_audio(src_path: str, dst_path: str, max_sec: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", str(SR), "-t", str(max_sec),
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpegでの音声抽出に失敗しました:\n{result.stderr[-800:]}")


# ---------- 分析 ----------
def estimate_key(chroma_mean: np.ndarray):
    best = (-2.0, 0, "major")
    for i in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            corr = np.corrcoef(chroma_mean, np.roll(prof, i))[0, 1]
            if corr > best[0]:
                best = (corr, i, mode)
    corr, tonic, mode = best
    return NOTE_NAMES[tonic], mode, float(corr)


def build_chord_templates():
    names, templates = [], []
    for i in range(12):
        maj = np.zeros(12)
        maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1
        names.append(NOTE_NAMES[i])
        templates.append(maj)
        mnr = np.zeros(12)
        mnr[[i, (i + 3) % 12, (i + 7) % 12]] = 1
        names.append(NOTE_NAMES[i] + "m")
        templates.append(mnr)
    t = np.array(templates)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    return names, t


def estimate_chords(chroma_sync: np.ndarray):
    names, templates = build_chord_templates()
    c = chroma_sync / (np.linalg.norm(chroma_sync, axis=0, keepdims=True) + 1e-9)
    idx = np.argmax(templates @ c, axis=0)
    seq = [names[i] for i in idx]
    # 連続する同じコードをまとめる
    compressed = [seq[0]]
    for ch in seq[1:]:
        if ch != compressed[-1]:
            compressed.append(ch)
    return seq, compressed


def analyze(wav_path: str, n_segments: int | None = None) -> dict:
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    duration = len(y) / sr

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    key_name, mode, key_conf = estimate_key(chroma.mean(axis=1))

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)

    n_frames = chroma.shape[1]
    beat_idx = librosa.util.fix_frames(beats, x_min=0, x_max=n_frames - 1)
    chroma_sync = librosa.util.sync(chroma, beat_idx, aggregate=np.median)
    chords_per_beat, chords = estimate_chords(chroma_sync)

    # 構成(セクション)分割: chroma + MFCC をビート同期して凝集クラスタリング
    feat = np.vstack([
        librosa.util.normalize(chroma, axis=1),
        librosa.util.normalize(mfcc, axis=1),
    ])
    feat_sync = librosa.util.sync(feat, beat_idx, aggregate=np.mean)
    k = n_segments or int(np.clip(round(duration / 25), 4, 12))
    k = min(k, feat_sync.shape[1])
    bounds = librosa.segment.agglomerative(feat_sync, k)
    bound_frames = beat_idx[np.clip(bounds, 0, len(beat_idx) - 1)]
    bound_times = list(librosa.frames_to_time(bound_frames, sr=sr, hop_length=HOP))
    bound_times[0] = 0.0
    bound_times.append(duration)

    sections = []
    for i in range(len(bound_times) - 1):
        s, e = bound_times[i], bound_times[i + 1]
        fs = librosa.time_to_frames(s, sr=sr, hop_length=HOP)
        fe = max(librosa.time_to_frames(e, sr=sr, hop_length=HOP), fs + 1)
        sections.append({
            "start": float(s),
            "end": float(e),
            "energy": float(rms[fs:fe].mean()),
        })
    max_e = max(s["energy"] for s in sections) or 1.0
    for s in sections:
        s["energy_rel"] = s["energy"] / max_e

    return {
        "y": y, "sr": sr, "duration": duration,
        "tempo": tempo, "beat_times": librosa.frames_to_time(beats, sr=sr, hop_length=HOP),
        "key": key_name, "mode": mode, "key_conf": key_conf,
        "chroma": chroma,
        "rms_mean": float(rms.mean()), "rms_std": float(rms.std()),
        "centroid_mean": float(centroid.mean()),
        "chords": chords, "chords_per_beat": chords_per_beat,
        "sections": sections,
    }


def top_progressions(chords: list[str], n: int = 4, top: int = 5):
    grams = [tuple(chords[i:i + n]) for i in range(len(chords) - n + 1)]
    return Counter(grams).most_common(top)


def rough_impression(r: dict) -> list[str]:
    """ルールベースのざっくりした印象タグ(あくまで目安)"""
    tags = []
    t = r["tempo"]
    tags.append("ゆったり" if t < 90 else "ミドルテンポ" if t < 125 else "アップテンポ")
    tags.append("明るめの調(長調)" if r["mode"] == "major" else "切なめ・暗めの調(短調)")
    c = r["centroid_mean"]
    tags.append("高域が華やか" if c > 3000 else "落ち着いた音色" if c < 1800 else "標準的な音の明るさ")
    dyn = r["rms_std"] / (r["rms_mean"] + 1e-9)
    tags.append("抑揚が大きい" if dyn > 0.6 else "抑揚は小さめ(一定の音圧)")
    return tags


# ---------- 歌詞(任意) ----------
def transcribe(wav_path: str, model_size: str = "small"):
    from faster_whisper import WhisperModel  # 遅延import(任意機能)
    model = WhisperModel(model_size, compute_type="int8")
    segments, info = model.transcribe(wav_path, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments)
    return text, info.language


# ---------- LLM解説 ----------
def fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def build_prompt(r: dict, title: str, lyrics: str | None, lyrics_lang: str | None) -> str:
    sec_lines = "\n".join(
        f"- {fmt_time(s['start'])}〜{fmt_time(s['end'])} 相対音量 {s['energy_rel']:.2f}"
        for s in r["sections"]
    )
    progs = top_progressions(r["chords"])
    prog_lines = "\n".join(f"- {' → '.join(p)} ({c}回)" for p, c in progs) or "- (検出なし)"
    lyric_part = ""
    if lyrics:
        lyric_part = f"\n## 歌詞の書き起こし(言語: {lyrics_lang}、誤認識を含む)\n{lyrics[:3000]}\n"
    return f"""あなたは音楽に詳しい解説者です。以下は音声分析ソフトによる「推定値」です。
推定には誤差があるので、断定しすぎず、分析結果から読み取れることを中心に、
この曲がどんな曲か(雰囲気、構成、盛り上がり方、和声の特徴、聴きどころ)を日本語で400〜600字で解説してください。
最後に「似た雰囲気のジャンル・アーティストの方向性」を一言添えてください。

曲名/ファイル名: {title}
長さ: {fmt_time(r['duration'])}
推定テンポ: {r['tempo']:.0f} BPM
推定キー: {r['key']} {r['mode']} (信頼度の目安 {r['key_conf']:.2f})
ざっくり印象タグ: {', '.join(rough_impression(r))}

## セクション(自動分割。音量は最大セクションを1とした相対値)
{sec_lines}

## 頻出するコード進行(推定)
{prog_lines}
{lyric_part}"""


def call_llm(prompt: str, api_key: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


# ---------- 可視化 ----------
def plot_waveform(y, sr, sections):
    step = max(len(y) // 4000, 1)
    ys = y[::step]
    ts = np.arange(len(ys)) * step / sr
    fig = go.Figure(go.Scatter(x=ts, y=ys, mode="lines", line=dict(width=1)))
    for s in sections[1:]:
        fig.add_vline(x=s["start"], line_dash="dot", line_color="red")
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="秒", title="波形(赤い点線がセクション境界)")
    return fig


def plot_chroma(chroma, sr):
    step = max(chroma.shape[1] // 800, 1)
    c = chroma[:, ::step]
    ts = np.arange(c.shape[1]) * step * HOP / sr
    fig = go.Figure(go.Heatmap(z=c, x=ts, y=NOTE_NAMES, colorscale="Viridis"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="秒", title="クロマグラム(どの音が鳴っているか)")
    return fig


def plot_sections(sections):
    labels = [f"{fmt_time(s['start'])}" for s in sections]
    fig = go.Figure(go.Bar(x=labels, y=[s["energy_rel"] for s in sections]))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="開始時刻", yaxis_title="相対音量",
                      title="セクションごとの盛り上がり")
    return fig


# ---------- UI ----------
def main():
    st.set_page_config(page_title="MV曲分析", page_icon="🎵", layout="wide")
    st.title("🎵 MV曲分析アプリ")
    st.caption("MV(動画)や音声をアップロードすると、テンポ・キー・構成・コード進行を推定し、AIが曲を解説します。")

    with st.sidebar:
        st.header("設定")
        max_sec = st.slider("解析する最大の長さ(秒)", 60, 600, 360, 30)
        use_whisper = st.checkbox("歌詞を書き起こして解説に使う(Whisper・重い)", value=False)
        whisper_size = st.selectbox("Whisperモデル", ["tiny", "base", "small", "medium"], index=2,
                                    disabled=not use_whisper)
        st.divider()
        st.subheader("AI解説(任意)")
        api_key = st.text_input("Anthropic APIキー", type="password",
                                value=os.environ.get("ANTHROPIC_API_KEY", ""))
        model = st.text_input("モデル名", value="claude-sonnet-5-5")

    up = st.file_uploader("MV・音声ファイル", type=["mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a", "flac", "ogg"])
    if not up:
        st.info("ファイルをアップロードしてください。(処理後、一時ファイルは削除されます)")
        return

    if st.button("分析する", type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "input" + os.path.splitext(up.name)[1])
            wav = os.path.join(tmp, "audio.wav")
            with open(src, "wb") as f:
                f.write(up.getbuffer())
            try:
                with st.status("分析中...", expanded=True) as status:
                    st.write("音声を抽出しています")
                    extract_audio(src, wav, max_sec)
                    st.write("テンポ・キー・構成・コードを推定しています")
                    result = analyze(wav)
                    lyrics, lang = None, None
                    if use_whisper:
                        st.write("歌詞を書き起こしています(時間がかかります)")
                        try:
                            lyrics, lang = transcribe(wav, whisper_size)
                        except ImportError:
                            st.warning("faster-whisper が未インストールのためスキップしました。")
                    status.update(label="分析完了", state="complete")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
                return
        st.session_state["result"] = result
        st.session_state["lyrics"] = (lyrics, lang)
        st.session_state["title"] = up.name
        st.session_state.pop("summary", None)

    r = st.session_state.get("result")
    if not r:
        return
    title = st.session_state["title"]
    lyrics, lang = st.session_state.get("lyrics", (None, None))

    # 概要
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("テンポ", f"{r['tempo']:.0f} BPM")
    c2.metric("キー(推定)", f"{r['key']} {'長調' if r['mode'] == 'major' else '短調'}")
    c3.metric("長さ", fmt_time(r["duration"]))
    c4.metric("セクション数", len(r["sections"]))
    st.write("**ざっくり印象:** " + " / ".join(rough_impression(r)))
    st.caption("※ すべて音声からの推定値です。特にコードやキーは誤りを含むことがあります。")

    # AI解説
    st.subheader("AIによる解説")
    prompt = build_prompt(r, title, lyrics, lang)
    if api_key:
        if st.button("AIに解説してもらう"):
            with st.spinner("解説を生成中..."):
                try:
                    st.session_state["summary"] = call_llm(prompt, api_key, model)
                except Exception as e:  # noqa: BLE001
                    st.error(f"LLM呼び出しに失敗しました: {e}")
        if "summary" in st.session_state:
            st.write(st.session_state["summary"])
    else:
        st.info("サイドバーにAPIキーを入れると、ここで解説を生成できます。キーなしでも、下のプロンプトを好きなAIに貼れば同じ解説が得られます。")
    with st.expander("LLMに渡すプロンプトを見る"):
        st.code(prompt, language="markdown")

    # 可視化
    st.subheader("構成と盛り上がり")
    st.plotly_chart(plot_waveform(r["y"], r["sr"], r["sections"]), use_container_width=True)
    st.plotly_chart(plot_sections(r["sections"]), use_container_width=True)
    st.dataframe(
        [{"区間": f"{fmt_time(s['start'])}〜{fmt_time(s['end'])}", "相対音量": round(s["energy_rel"], 2)}
         for s in r["sections"]],
        use_container_width=True,
    )

    st.subheader("和声")
    progs = top_progressions(r["chords"])
    for p, cnt in progs:
        st.write(f"- {' → '.join(p)}  ({cnt}回)")
    st.plotly_chart(plot_chroma(r["chroma"], r["sr"]), use_container_width=True)
    with st.expander("推定コード列(重複をまとめたもの)"):
        st.write(" | ".join(r["chords"][:200]))

    if lyrics is not None:
        st.subheader("歌詞")
        st.write(f"言語: {lang} / 書き起こし文字数: {len(lyrics)}(全文表示は著作権に配慮して省略。解説には利用されます)")


if __name__ == "__main__":
    main()
