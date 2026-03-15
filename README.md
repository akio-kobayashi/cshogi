# cshogi: Pythonのための高速な将棋ライブラリ

`cshogi`は、盤面の管理、合法手の生成、指し手の検証、USIプロトコルのサポート、そして機械学習でよく使われるフォーマットへの対応など、多くの機能を提供する高速なPython将棋ライブラリです。

## インストール

お使いの環境に対応したコンパイル済みのwheelが利用可能な場合、PyPIから簡単にインストールできます。

```bash
pip install cshogi
```

Webインターフェースも使いたい場合は、追加の依存関係をインストールしてください。
```bash
pip install cshogi[web]
```

### このフォークをインストールする

`shogi_ai` や `DeepLearningShogi` からこのフォーク版 `cshogi` を使う場合は、PyPI 版ではなく GitHub 上のこのリポジトリを直接インストールしてください。

```bash
pip install --upgrade "git+https://github.com/akio-kobayashi/cshogi.git"
```

ローカルで clone 済みの作業ツリーをそのまま使う場合は、編集可能インストールでも構いません。

```bash
git clone https://github.com/akio-kobayashi/cshogi.git
cd cshogi
pip install -e .
```

### このフォークをアップデートする

GitHub から直接インストールした場合は、同じコマンドを再実行すれば更新できます。

```bash
pip install --upgrade "git+https://github.com/akio-kobayashi/cshogi.git"
```

ローカル clone を `pip install -e .` で使っている場合は、リポジトリを更新してから再インストールしてください。

```bash
git pull
pip install -e .
```

## ソースからのビルド (Linux/Ubuntuの場合)

もしコンパイル済みのwheelが利用できない場合や、最新のソースコードからビルドしたい場合は、手動でビルドすることができます。このプロジェクトは`setuptools`と`Cython`を使ってC++拡張機能をコンパイルします。

**1. 必要なツールをインストールする**

C++コンパイラ、Python開発用ヘッダ、そしてpipが必要です。

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

**2. リポジトリをクローンする**

```bash
git clone https://github.com/TadaoYamaoka/cshogi.git
cd cshogi
```

**3. ビルドに必要なPythonパッケージをインストールする**

拡張機能のビルドに必要なPythonパッケージをインストールします。

```bash
pip install cython numpy
```

**4. cshogiをビルドしてインストールする**

pipを使ってライブラリをコンパイルし、インストールします。

```bash
pip install .
```
このコマンドは`setup.py`を使い、Cython拡張機能をビルドして、あなたのPython環境にパッケージをインストールします。

## 主な機能

*   Python 3.6+ と Cython 0.29+ をサポート
*   IPython/Jupyter Notebook上での盤面表示
*   指し手を進める(`push`)、元に戻す(`pop`)機能
*   テキストベースでの盤面表示
*   王手、詰み、引き分けの判定（千日手や入玉宣言も含む）
*   USI形式およびCSA形式の指し手に対応
*   AperyやYaneuraOuで使われる圧縮された局面フォーマットの読み書き
*   USIプロトコルに対応した将棋エンジンとの通信

## 特徴量抽出 (`FeatureExtractor`)

`cshogi`には、将棋の盤面から機械学習で利用しやすい多様な特徴量を抽出するための`FeatureExtractor`クラスが含まれています。

### 使い方

`FeatureExtractor`は`cshogi.Board`オブジェクトを元に初期化します。様々なメソッドを使って、盤面の状態、評価値に関連する情報、直前の指し手や合法手に関する統計などを取得できます。

以下は基本的な使用例です。

```python
import cshogi
from cshogi import FeatureExtractor

# 盤面を初期化
board = cshogi.Board()

# 現在の盤面に対するExtractorを作成
extractor = FeatureExtractor(board)

# --- A群: 現在の盤面に関する特徴量 ---
board_layout = extractor.get_board_layout_2d() # 盤面レイアウト (9x9x28のTensor)
kings = extractor.get_king_positions()        # 両方の玉の位置
hand_pieces = extractor.get_hand_pieces()     # 両方の持ち駒
is_check = extractor.is_check()               # 現手番の玉に王手がかかっているか
legal_moves_analysis = extractor.analyze_legal_moves() # 合法手の分析

print("盤面レイアウトのShape:", board_layout.shape)
print("玉の位置 (先手, 後手):", kings)
print("持ち駒:", hand_pieces)
print("王手がかかっているか？:", is_check)
print("合法手の分析:", legal_moves_analysis)


# --- C群: 直前の指し手に関する特徴量 ---
# この機能を使うには、指し手を指す「前」の盤面が必要です

# 例として、初手から2手進める
initial_board = cshogi.Board()
move1 = cshogi.Move.from_usi('7g7f')
initial_board.push(move1)

previous_board = initial_board.copy() # 2手目を指す前の盤面をコピー
move2 = cshogi.Move.from_usi('3c3d')
initial_board.push(move2)             # 2手目を指す

# `get_last_move_features`はstatic methodなので、クラスから直接呼び出す
# 第一引数: 指す前の盤面, 第二引数: 指した手
last_move_features = FeatureExtractor.get_last_move_features(previous_board, move2)
print("\n指し手 '3c3d' の特徴量:", last_move_features)


# --- E群: 対局全体に関する特徴量 ---
# これらの特徴量を取得するには、Eloや勝敗などの外部情報が必要です
game_metadata = {
    'total_moves': 120,
    'winner': cshogi.BLACK,
    'black_elo': 2800,
    'white_elo': 2750
}
# この機能もstatic method
game_features = FeatureExtractor.get_game_state_features(initial_board, game_metadata)
print("\n対局全体に関する特徴量:", game_features)
```

## 謝辞

cshogiの高性能な機能の多くは、将棋ソフト`Apery`のソースコードを利用しています。

## ライセンス

cshogiはGPLv3ライセンスの下で公開されています。詳細は`LICENSE`ファイルをご覧ください。
