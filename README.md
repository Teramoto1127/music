# MV曲分析アプリ(MVP)

MV(動画)や音声をアップロードして、テンポ・キー・構成・コード進行を推定し、AIが「どんな曲か」を解説するアプリです。

## セットアップ

1. ffmpeg をインストール(Mac: `brew install ffmpeg` / Windows: `winget install ffmpeg`)
2. 依存パッケージ: `pip install -r requirements.txt`
3. (任意) AI解説: 環境変数 `ANTHROPIC_API_KEY` を設定、またはアプリのサイドバーに入力
4. (任意) 歌詞の書き起こし: `pip install faster-whisper`
5. 起動: `streamlit run app.py`

## 2つのモード

サイドバーで切り替えます。

- **ファイルを分析(音声解析)**: 手元のMV/音声ファイルからテンポ・キー・構成・コードを推定し、AIが解説
- **URL・曲名から解説(AI)**: YouTube(YouTube Music)のURLまたは「アーティスト名 曲名」を入力。音声は解析せず、YouTubeの情報とAIの知識で解説(テンポやキーなどの数値は出ません)
  - URLならAPIキーなしでタイトル・チャンネル名を取得(oEmbed)
  - `YOUTUBE_API_KEY`(YouTube Data API v3)を設定すると、説明文・タグ・公開日・再生数も使えます

## 仕組み(ファイル分析モード)

動画 → ffmpegで音声抽出 → librosaでテンポ/キー/クロマ/構成/コード推定 → 結果をプロンプト化 → LLMが解説

## 次にやるとよいこと

- 音源分離(demucs)でボーカルを分けて、歌詞認識やメロディ分析の精度を上げる
- basic-pitch でメロディをMIDI化 → music21 で旋律・和声分析
- ムード/ジャンル判定を学習済みモデル(essentia, CLAP)に置き換える
- 曲同士の類似度検索
- 動画フレームを抜き出して、MVの映像面の解説も追加する
