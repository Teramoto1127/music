"""YouTube URL または 曲名から、AIが曲を解説するモード。

音声は解析しない。YouTubeのメタデータ(タイトル・チャンネル・説明文など)と、
AIが持っている知識をもとに解説する。テンポやキーなどの数値は出さない。
"""
import os
import re

import requests
import streamlit as st

YT_URL_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")
YT_API = "https://www.googleapis.com/youtube/v3/videos"
OEMBED = "https://www.youtube.com/oembed"
TIMEOUT = 10


def extract_video_id(text: str) -> str | None:
    """YouTube / YouTube Music のURLから動画IDを取り出す。URLでなければ None。"""
    if "youtube.com" not in text and "youtu.be" not in text:
        return None
    m = YT_URL_RE.search(text)
    return m.group(1) if m else None


def fetch_oembed(video_id: str) -> dict:
    """APIキー不要。タイトルとチャンネル名だけ取れる。"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    r = requests.get(OEMBED, params={"url": url, "format": "json"}, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    return {"title": d.get("title", ""), "channel": d.get("author_name", "")}


def fetch_api(video_id: str, api_key: str) -> dict:
    """YouTube Data API v3。説明文・タグ・公開日・再生数なども取れる。"""
    r = requests.get(
        YT_API,
        params={"part": "snippet,contentDetails,statistics", "id": video_id, "key": api_key},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        raise ValueError("動画が見つかりませんでした(非公開・削除済み、またはIDの誤りの可能性があります)。")
    it = items[0]
    sn, cd, stt = it.get("snippet", {}), it.get("contentDetails", {}), it.get("statistics", {})
    return {
        "title": sn.get("title", ""),
        "channel": sn.get("channelTitle", ""),
        "description": sn.get("description", ""),
        "tags": sn.get("tags", []),
        "published": sn.get("publishedAt", "")[:10],
        "duration": cd.get("duration", ""),
        "views": stt.get("viewCount", ""),
    }


def get_metadata(video_id: str, yt_api_key: str) -> dict:
    """APIキーがあればData API、なければoEmbedで取得する。"""
    if yt_api_key:
        return fetch_api(video_id, yt_api_key)
    return fetch_oembed(video_id)


def build_prompt(query: str, meta: dict | None) -> str:
    if meta:
        lines = [f"- タイトル: {meta.get('title', '')}", f"- チャンネル: {meta.get('channel', '')}"]
        if meta.get("published"):
            lines.append(f"- 公開日: {meta['published']}")
        if meta.get("views"):
            lines.append(f"- 再生数: {meta['views']}")
        if meta.get("tags"):
            lines.append(f"- タグ: {', '.join(meta['tags'][:15])}")
        if meta.get("description"):
            lines.append(f"- 説明文(先頭のみ):\n{meta['description'][:1500]}")
        info = "## YouTubeから取得した情報\n" + "\n".join(lines)
    else:
        info = f"## 入力された曲名\n{query}"

    return f"""あなたは音楽に詳しい解説者です。以下の曲について、日本語で400〜600字で解説してください。

重要なルール:
- あなたはこの曲の音声を聴いていません。テンポ(BPM)やキーなどの数値は、確かな知識がない限り書かないでください。
- 事実(知っていること)と推測(タイトルやジャンルからの想像)を区別して書いてください。
- よく知らない曲・アーティストの場合は、無理に作らず「情報が少ない」と正直に書き、分かる範囲(ジャンルの傾向など)にとどめてください。
- 歌詞は引用しないでください。テーマや雰囲気の要約にとどめてください。

解説に含める観点: どんなジャンル・雰囲気の曲か / アーティストや背景 / 聴きどころ / 似た方向性のジャンルやアーティスト

{info}
"""


def render_metadata_mode(api_key: str, model: str, call_llm) -> None:
    st.subheader("URL・曲名から解説")
    st.caption("音声は解析しません。YouTubeの情報(タイトル・説明文など)とAIの知識で解説します。テンポやキーなどの数値は出ません。")

    query = st.text_input("YouTube / YouTube Music のURL、または「アーティスト名 曲名」")
    with st.expander("YouTube Data APIキー(任意・入れると説明文やタグも使えます)"):
        yt_key = st.text_input("YouTube Data API v3 キー", type="password",
                               value=os.environ.get("YOUTUBE_API_KEY", ""))
        st.caption("未入力でもURLならタイトルとチャンネル名は取得できます。")

    if not st.button("解説する", type="primary"):
        return
    query = query.strip()
    if not query:
        st.warning("URLまたは曲名を入力してください。")
        return

    meta = None
    video_id = extract_video_id(query)
    if video_id:
        try:
            with st.spinner("YouTubeの情報を取得中..."):
                meta = get_metadata(video_id, yt_key)
        except (requests.RequestException, ValueError) as e:
            st.error(f"YouTubeの情報を取得できませんでした: {e}")
            return
        with st.expander("取得した情報", expanded=True):
            st.write(f"**{meta['title']}** / {meta['channel']}")
            if meta.get("published"):
                st.write(f"公開日: {meta['published']}")
    elif "http" in query:
        st.error("YouTube以外のURLには対応していません。曲名を入力してください。")
        return

    prompt = build_prompt(query, meta)
    if api_key:
        with st.spinner("解説を生成中..."):
            try:
                st.write(call_llm(prompt, api_key, model))
            except Exception as e:  # noqa: BLE001
                st.error(f"LLM呼び出しに失敗しました: {e}")
        st.caption("※ 音声を聴いて解析した結果ではありません。内容は不正確な場合があります。")
    else:
        st.info("サイドバーにAnthropic APIキーを入れると、ここで解説を生成できます。下のプロンプトを好きなAIに貼っても同じ解説が得られます。")
    with st.expander("AIに渡すプロンプトを見る"):
        st.code(prompt, language="markdown")
