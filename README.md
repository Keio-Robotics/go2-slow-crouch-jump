# go2-slow-crouch-jump

Unitree Go2 の内蔵ジャンプ（FrontJump）を実機で記録し、**しゃがみ部分だけをゆっくり（数秒〜10分）に引き伸ばしてから、記録どおりに跳ばす**ためのコード一式です。低レベル制御（`/lowcmd`）で、記録した関節角度・角速度・トルクをそのまま再生します。

- 内蔵ジャンプは「素早くしゃがむ(0.23 s) → 最下点で約0.8 s 静止 → 踏み切り」という流れです。最下点で止まっているので、しゃがみだけを引き伸ばしても、踏み切りから後ろは記録どおりに再生できます。
- 2026-11 に実機で確認できたこと:
  - しゃがみ 2.3 秒からのジャンプを、R1+A を押すたびに繰り返す（2回）
  - しゃがみ 10 分からのジャンプ（1回）

> **注意**: 記録どおりの踏み切りを、フィードバックなしで（決め打ちで）再生します。転倒する可能性があります。必ず「安全」の節を読んでから実行してください。

## ファイル

| ファイル | どこで動かすか | 役割 |
|---|---|---|
| `go2_record_jump.py` | Jetson | `/lowstate` と `/sportmodestate` を記録する。`--jump` で BalanceStand → FrontJump を送る。`--check-mode` でモード確認 |
| `go2_crouch_replay.py` | Jetson | 吊り下げ試験用。しゃがみ区間だけを引き伸ばして再生し、元の姿勢に戻る（ジャンプしない） |
| `go2_jump_replay.py` | Jetson | 床の上での本番用。立ち上がり → 待機 → R1+A でゆっくりしゃがみ＋記録ジャンプ → 立ち姿勢に戻る（繰り返し可） |
| `crouch_plan.py` / `jump_plan.py` | 両方 | 目標値の計画（numpy のみ）。実機ノードとシミュレーションで共通 |
| `release_sport_mode.py` | Jetson | 内蔵制御を解除する（**この瞬間に脚の力が抜ける**） |
| `standdown.py` | Jetson | 内蔵制御で伏せさせる（解除前に使う） |
| `motor_temps.py` | Jetson | 12個のモーター温度とバッテリー残量を表示する |
| `go2_crc.py` | Jetson | LowCmd の CRC |
| `sim_replay_jump.py` | PC | 記録を MuJoCo でそのまま／引き伸ばして再生する |
| `sim_check_crouch_plan.py` / `sim_check_jump_plan.py` | PC | 実機ノードと同じ計画を MuJoCo で検証する |
| `data/jump01.npz` / `.json` | — | 実機での FrontJump の記録（約500Hz、12関節の q/dq/tau_est、接地力、IMU） |
| `data/jump01_replay.npz` | — | `jump_plan.extract_jump` で作った再生用の軌道（しゃがみ開始〜着地後 2.5 s） |
| `data/crouch01.npz` | — | `crouch_plan.extract_crouch` で作った、しゃがみ区間だけの軌道（ジャンプを含まない） |

関節の並びは、すべて Unitree SDK の順（FR, FL, RR, RL × hip, thigh, calf）です。

## 必要なもの

- **Go2 EDU**（低レベル制御ができる機体）と、Jetson などの搭載PC。Go2 の内部ネットワーク（192.168.123.x）につながっていること
- Jetson 側:
  - ROS 2 Humble と [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)（`unitree_go`、`unitree_api` のメッセージ）
  - CycloneDDS の設定
  - Python 3 と numpy
- PC 側（シミュレーションだけ）: `pip install mujoco numpy imageio imageio-ffmpeg`。MuJoCo Menagerie の Go2 モデルも必要です:
  ```bash
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
  ```
  このフォルダの直下に `mujoco_menagerie/` を置くか、`GO2_SCENE=/path/to/unitree_go2/scene.xml` を指定してください。

## 安全

- 停止方法（ノードの実行中）:
  - **L2+B**: いつでも、すぐにダンピング（kp=0, kd=5）。一度入ると戻らない
  - **L1**: 立っているときに押すと、座ってから終了する
  - Ctrl-C / SIGTERM: ダンピング
- 監視（基準を超えるとダンピング）:
  - テレメトリの途絶
  - 胴体の傾き 34° 超
  - 関節速度 8 rad/s 超（踏み切り中は 30 rad/s 超）
  - 目標とのずれ 0.6 rad 超（踏み切り中は 1.2 rad 超）
  - **モーター温度 75°C 以上**（65°C で警告）
- 内蔵制御が動いたまま `--arm` で起動すると、起動を拒否します。
- 前方 2 m・周囲 1 m に人や物がない、平らで滑らない床で行ってください。
- Go2 と Go2W は、搭載PC の IP アドレスが同じです。車輪のモーターが回っていれば起動を拒否します。さらに `--expected-hostname <Jetsonのホスト名>` を付けると、それ以外のマシンでは動かなくなります。
- 低レベル制御のあとに内蔵制御に戻すには、ロボットを再起動するのが確実です。

## 使い方

### 0. Jetson に配置する

```bash
scp -r go2-slow-crouch-jump <user>@<jetson>:~/
ssh <user>@<jetson>
source <unitree_ros2>/setup.sh        # CycloneDDS の設定込み
cd ~/go2-slow-crouch-jump
```

以下のコマンドは、すべて Jetson 上のこのフォルダで実行します（シミュレーションを除く）。`--arm` を付けない限り、実機ノードはドライラン（何も送らない）です。

### 1. 状態を確認する（何も送らない）

```bash
python3 go2_record_jump.py --check-mode         # 内蔵制御のモード名（解除済みなら空）
python3 go2_record_jump.py --duration 3 --out dry_record   # 3秒記録するだけ
python3 motor_temps.py
```

### 2.（任意）自分の機体で FrontJump を記録し直す

**別の機体で記録した軌道を使うと、長時間しゃがみで特定の関節が過熱しやすくなります**（「既知の問題」を参照）。できれば、使う機体で記録し直してください。

```bash
# 内蔵制御で立っている状態で。BalanceStand → 3秒後に FrontJump → 4秒記録
python3 go2_record_jump.py --jump --out data/jump02
python3 -c "from jump_plan import extract_jump; print(extract_jump('data/jump02.npz', 'data/jump02_replay.npz'))"
python3 -c "from crouch_plan import extract_crouch; print(extract_crouch('data/jump02.npz', 'data/crouch02.npz'))"
```

### 3.（PC）シミュレーションで確かめる

```bash
python3 sim_replay_jump.py data/jump01.npz                 # 記録どおり
python3 sim_replay_jump.py data/jump01.npz --stretch 10    # しゃがみ10倍
python3 sim_check_jump_plan.py --jumps 3                   # 伏せ→立つ→3回跳ぶ→座る（実機ノードと同じ計画）
python3 sim_check_jump_plan.py --video jump.mp4
```

### 4. 吊り下げて、しゃがみだけを試す

ロボットを吊り下げ、しゃがんでも足が床に着かない高さにします。

```bash
python3 go2_crouch_replay.py                   # ドライラン
python3 release_sport_mode.py                  # 脚の力が抜ける
python3 go2_crouch_replay.py --arm             # 11.7秒で終了し、ダンピング
```

### 5. 床の上でジャンプを再生する

```bash
python3 standdown.py                           # 内蔵制御で伏せる
python3 release_sport_mode.py
python3 go2_record_jump.py --check-mode        # name が空であることを確認
python3 go2_jump_replay.py --arm               # 3秒で立ち上がって待機
```

- 待機中に **R1+A** を押すと、しゃがみ（既定 2.3 秒）→ 記録ジャンプ → 立ち姿勢に戻ります。
- もう一度跳ぶときは、R1+A を一度離してから押し直します（前のジャンプから2秒以上あける）。
- 何もしないまま 300 秒たつと、自動で座ります。

主なオプション:

| オプション | 意味 |
|---|---|
| `--crouch-secs 600` | しゃがみに 600 秒（10分）かける |
| `--stretch 10` | しゃがみを記録の何倍に引き伸ばすか（`--crouch-secs` があればそちらが優先） |
| `--traj data/jump02_replay.npz` | 使う軌道ファイル |
| `--ff-scale 1.0` | 上乗せする記録トルクの倍率 |
| `--max-wait 300` | 待機の上限（秒） |
| `--log run.npy` | 指令と状態の記録先 |

SSH が切れても止まらないように起動する例:

```bash
setsid bash -c "source <unitree_ros2>/setup.sh; cd ~/go2-slow-crouch-jump; \
  timeout 3600 python3 go2_jump_replay.py --arm --crouch-secs 600 > run.log 2>&1" </dev/null >/dev/null 2>&1 &
tail -f run.log
```

## 既知の問題

- **長時間しゃがみでの過熱**: 記録したときの立ち姿勢が左右非対称だと、脚同士が押し合い、特定の関節がトルクを出し続けます。
  - 10分しゃがみで、記録した機体では FL の太もも関節が 74°C まで上がりました。
  - 別の機体に同じ軌道を使ったところ、RL の股関節が平均 9.2 Nm を出し続け、約4分40秒で 75°C に達して停止しました。
  - 対策案: 使う機体で記録し直す、軌道を左右対称にする、長時間しゃがみ中は股関節のゲインを下げてトルクの上乗せをやめる。
- `LowState.fan_frequency` は、内蔵制御中でも低レベル制御中でも 0 で、ファンの状態は読めませんでした。
- FL の足の接地力センサーは、他の足の約半分の値しか出ないことがあります。
- シミュレーション（MuJoCo Menagerie の Go2）では、モーターのトルク-速度特性と通信遅延は再現していません。実機の推定トルクは、モデルの上限（股・太もも 23.7 Nm、ふくらはぎ 45.4 Nm）を超えることがあります。
