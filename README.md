# 環境開発
## RealSense
RealSenseのSDK入れる

https://github.com/IntelRealSense/librealsense

RealSenseの設置は砂場の真上で，カメラ向きは真下．

## ライブラリインストール
Python 3.10.15
```
pip install -r requirements.txt
```

# 実行
```
python effect_two.py
```
RealSenseデバイスを使用して、カメラ映像と深度データを取得します。実行中に、１つのウィンドウに2つのフレームがが表示されます:
- RGB画像と手検出: 手の位置がMediapipeを使って検出され、赤い点で表示されます。
- 深度データのカラーマップ: 深度データがカラー表示されます。

`s`キーを押すと、現在の深度データをもとに3Dポイントクラウドを保存し、演奏準備が整います。このとき、次のメッセージが表示されます:
```
3Dデータが保存されました
```
表示されたら演奏準備が整います。あとは手で叩いて楽しむだけです。


# 開発ログ
- create3d.py
    - 3Dデータの保存ができた。背景差分では手の位置検出うまくいかず。

- create_sound.py
    - スケールおかしいけど機能はできた

- viss.py
    - ビジュアル化完了

- touch.py
    - 機能は完全に満たした。高速化必須

- base_line.py
    - 可視化機能を削除

- hand_detect.py
    - 手の検知を強化

- effect_two.py
    - 高さによって音が変える仕組みを追加

- model_accuracy.py
    - 前処理など追加

- multi_hands.py
    - 複数の手に対応

- good_sound.py
    - 音の体験を改善
- palam_top.py
    - パラメータを上部にもってくる
- hand_def.py（最新版）
    - 手の検出を関数として取り出した