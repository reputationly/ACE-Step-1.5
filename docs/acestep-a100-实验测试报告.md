# ACE-Step-1.5 A100 实验测试报告(文生音乐)

> 平台:鲲鹏920 ARM aarch64 + A100 PCIE 40G(sm_80,无 NVLink)· 测试机 dev-gpustack-a100-0019
> 驱动 570.86.10 / CUDA 12.8 → **锁 cu128**(cu130 需驱动≥580,本机跑不了)
> 镜像:`crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com/reputationly/acestep:arm64-a100-latest`
> (FROM `arronlee/lightx2v:arm64-cu128-a100-base`,torch 2.11 cu128,自带 flash_attn2 sm_80)
> harness:`scripts/smoke/smoke_acestep_a100.sh`(矩阵)+ `scripts/smoke/stress_acestep.sh`(单容器压测)
> 权重:`/nfs-models/wuhanjisuan894/vllm-omni-speech/ACE-Step-1.5/`(挂容器 `/app/checkpoints`)
> 日期:2026-07-18 起(滚动填充)

> 方法论照搬 `LightX2V/docs/Wan2.2-{I2V,S2V}-实验测试报告.md`:结论先行 → 矩阵 → 根因 → 压测找天花板 → 质量确认 → 复现 → 速查。
> 测试纪律:**热态稳态**(连发丢首张取均值,冷态含加载虚高);**安静宿主**(`docker rm -f` 清场再测);容器 `--memory=240g`;**单容器复用扫压测**(加载 ~2min/次,别每档重载)。

---

## 0. 结论先行

1. **生产默认 = `acestep-v15-xl-turbo`(4B DiT)+ `acestep-5Hz-lm-4B`**;备选 turbo(2B,省 ~5G 显存)。XL 音质更饱满、代价近乎为零(+5G 显存、速度基本不变),故默认取音质。
2. 单卡 **1 副本**;4×A100 节点 = **4 副本**(或留 1 卡机动 = 3);**时长上限 = 600s 官方上限内无显存/截断瓶颈**(显存不随时长涨)。
3. 每实例吞吐(workers=1 串行,30s×2/请求 ~10s)≈ **6 请求/min**;**RTF ~0.08–0.17**(比实时快 6–12×)。
4. 判死/踩坑:**锁 cu128**(驱动 570=CUDA12.8,cu130 跑不了);lyrics=`[inst]` 出器乐、填真实歌词才唱;**CLAP 测贴合度不测保真**,DiT 选型靠"近乎免费 + 耳朵"定,不被 CLAP 主导;镜像 baked `HF_HUB_OFFLINE=1`(打分需 `-e HF_HUB_OFFLINE=0` 覆盖)。

---

## 1. 已有权重清单

| 类别 | 目录 | 大小 | 说明 |
|---|---|---|---|
| DiT | `acestep-v15-turbo` | 4.5G | 2B DiT,8步,默认 |
| DiT | `acestep-v15-turbo-shift3` | 4.5G | 2B DiT,shift=3 变体 |
| DiT | `acestep-v15-xl-turbo` | ~9G | **4B DiT,8步,音质 Very High**(下载中) |
| DiT | `acestep-v15-xl-sft` / `-xl-base` | ~9G | 已下但**本轮不测**(sft 50步/base 支持 extract/lego/complete) |
| LM | `acestep-5Hz-lm-1.7B` | 3.5G | 5Hz 语义 LM |
| LM | `acestep-5Hz-lm-4B` | 7.9G | 5Hz 语义 LM(最大) |
| 组件 | `vae` / `Qwen3-Embedding-0.6B` | 322M / 1.2G | VAE + 文本编码器(主仓) |

---

## 2. P1 — 模型 A/B 矩阵(30s,batch=2,steps=8,单卡)

> 每组换 `ACESTEP_CONFIG_PATH`/`ACESTEP_LM_MODEL_PATH` 重起容器一次。发 t2m-inst + t2m-vocal(zh/en)。

| # | DiT | LM | 加载(冷) | 生成(热) | 峰值显存 | RTF | 产物 | 听感 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| M1 | turbo(2B) | 1.7B | — | — | ~13.7G(已测) | — | — | — | |
| M2 | turbo(2B) | 4B | ~2min | ~10s/2条 | ~19.5G(已测) | — | 470K ✓ | — | ✅ 已测 |
| M3 | turbo-shift3 | 4B | — | — | ~19.4G | — | — | — | 并行跑,与 turbo 同量级 |
| M4 | **xl-turbo(4B)** | 4B | ~2min | ~10s/2条(热) | **23.8–26.5G** | — | 470K ✓ | 略优(耳朵) | ✅ **选定** |

**结论**:LM 1.7B→4B 显存 +6G(13.7→19.5G,turbo);turbo→xl-turbo DiT +5G(19.5→26.5G),速度几乎不变 → **XL 音质升级近乎白嫖**。

### 2.1 CLAP 客观 A/B(objective,`ab_clap.sh`,N=10,同 seed)

CLAP(`laion/larger_clap_music_and_speech`)音频↔caption 贴合度,turbo(A)vs xl-turbo(B):

| prompt | A turbo | B xl | Δ(B-A) | | prompt | A turbo | B xl | Δ(B-A) |
|---|---|---|---|---|---|---|---|---|
| pop_rock | 0.342 | 0.426 | +0.084 | | jazz | 0.379 | 0.354 | -0.024 |
| lofi | 0.624 | 0.367 | -0.257 | | trap | 0.249 | 0.186 | -0.064 |
| ballad | 0.320 | 0.504 | +0.184 | | cinematic | 0.207 | 0.253 | +0.046 |
| edm | 0.513 | 0.372 | -0.141 | | synthwave | 0.383 | 0.484 | +0.101 |
| folk | 0.459 | 0.351 | -0.108 | | metal | 0.460 | 0.351 | -0.110 |

均值 A=0.394 / B=0.365,Δ=-0.029;**paired Wilcoxon p=0.49(无显著差异)**,B>A 4/10。

**解读(重要)**:此结果**不能当"turbo 更好"采信**——① CLAP 测语义**贴合度**不测**保真**,而 XL 的增益恰是保真;② N=10 功效不足,p=0.49 是"测不出"非"相等";③ 单个 lofi 异常值(-0.26)主导均值,逐条方差极大。故 DiT 选型**不由 CLAP 定**,由"XL 近乎免费 + 耳朵边际优势"定 → xl-turbo。要客观判音质需 FAD(参考集,后置)。

---

## 3. P2 — 时长压测(turbo 2B + 4B LM,batch=2,steps=8,单卡,已测 2026-07-18)

> `stress_acestep.sh` 单容器循环。music 官方上限 600s。

| 时长 | 生成(热) | RTF(生成/音频) | 峰值显存 | 产物 | 时长对齐 | 状态 |
|---|---|---|---|---|---|---|
| 30s | 127s(⚠️含~2min加载) | — | 19385 MiB | 480813B | 30s ✓ | ✅ |
| 60s | 20s | 0.33 | 19387 MiB | 960813B | 60s ✓ | ✅ |
| 120s | 30s | 0.25 | 21031 MiB | 1920813B | 120s ✓ | ✅ |
| 240s | 45s | 0.19 | 20729 MiB | 3840813B | 240s ✓ | ✅ |
| 480s | 75s | 0.16 | 21029 MiB | 7680813B | 480s ✓ | ✅ |
| 600s | 100s | 0.17 | 21277 MiB | 9600813B | 600s ✓ | ✅ |

host:MemAvailable 241.7G / Shmem 97M(零内存压力,非大户)。

### xl-turbo(4B DiT)+ 4B LM 对照(同 seed=42,已测 2026-07-18)

| 时长 | 生成(热) | 峰值显存 | vs turbo Δ显存 | 对齐 |
|---|---|---|---|---|
| 60s | 20s | 23799 MiB | +4.4G | ✓ |
| 120s | 35s | 24839 MiB | +3.8G | ✓ |
| 240s | 45s | 24857 MiB | +4.2G | ✓ |
| 480s | 85s | 25809 MiB | +4.8G | ✓ |
| 600s | 100s | **26497 MiB** | +5.2G | ✓ |

**turbo vs xl-turbo 结论**:XL 4B DiT 仅 **+5G 显存**(600s 峰值 26.5G/40G,余 13G)、**速度几乎不变**(DiT 扩散 8 步占时间小头,LM CoT+VAE 是大头)→ **音质升级近乎白嫖**。显存同样不随时长涨。**是否采用取决于 A/B 听感**(Very High vs High 值不值 +5G)。

**结论(与视频报告相反)**:
- **显存几乎不随时长涨**:30s→600s(20×)仅 19.4G→21.3G(+1.9G)→ **时长无显存瓶颈**,600s(官方上限)单卡 ~21G 随便跑。音乐 latent 远比视频便宜。
- **速度**:600s(10min)曲 100s 出;batch=2 等效 1200s 音频/100s → RTF ~0.08,**比实时快 6-12×**。
- **时长精确对齐**到 600s,无截断/提前结束。
- 与视频对照:i2v 时长瓶颈是显存、flf2v 是质量;**music 在 600s 内两者都不触**。剩余唯一变量 = 长曲**听感连贯性**(需耳朵)。

---

## 4. P3 — 步数 / 批量

| steps | 生成(热) | 峰值显存 | 听感(降质?) |
|---|---|---|---|
| 4 | — | — | — |
| 8(默认) | — | — | — |
| 16 | — | — | — |

| batch | 生成(热) | 峰值显存 | 单条均摊 |
|---|---|---|---|
| 1 | — | — | — |
| 2(默认) | ~10s | ~19.5G | ~5s |
| 4 | — | — | — |

---

## 5. P4 — 任务面

| 任务 | 输入 | 生成 | 峰值显存 | 产物/听感 | 状态 |
|---|---|---|---|---|---|
| t2m-inst | 纯文本 [inst] | — | — | — | |
| t2m-vocal-zh | 中文歌词 | — | — | — | |
| t2m-vocal-en | 英文歌词 | — | — | — | |
| t2m-vocal-ja | 日文歌词 | — | — | — | |
| cover | 参考音频 | — | — | — | |
| repaint | 源音频+区间 | — | — | — | |

---

## 6. P5 — 并发 / 吞吐(sizing 关键)

> workers=1 内存队列串行。并发提交 N,测吞吐/延迟/背压(队列满 429)。

| 并发 | 完成总时长 | 吞吐(条/min) | 尾延迟 | 队列行为 |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

**根因待答**:workers=1 串行 → 每实例吞吐 ≈ 1/生成时长?→ gpustack 每卡 1 实例的 QPS 上限。

---

## 7. P6 — 崩溃边界(门面前置校验用)

| 输入 | 预期 | 实测 |
|---|---|---|
| 歌词超长(> ? 字) | 优雅 failed 或截断 | — |
| duration > 600 | 拒绝或钳制 | — |
| 空 prompt + 空 lyrics | ? | — |
| 非法 task_type | 400 | — |

---

## 8. 复现命令

```bash
REG=crpi-xzr81d0490mc3794.cn-shanghai.personal.cr.aliyuncs.com
IMG=$REG/reputationly/acestep:arm64-a100-latest
CK=/nfs-models/wuhanjisuan894/vllm-omni-speech/ACE-Step-1.5

# P1 模型 A/B(矩阵 harness,每个模型跑一次)
for DIT in acestep-v15-turbo acestep-v15-turbo-shift3 acestep-v15-xl-turbo; do
  IMAGE="$IMG" CKPT_DIR="$CK" GPUS='"device=0"' CONFIG_PATH="$DIT" LM_MODEL=acestep-5Hz-lm-4B \
    bash /root/smoke_acestep_a100.sh
done

# P2/P3 时长×步数×批量压测(单容器复用)
IMAGE="$IMG" CKPT_DIR="$CK" GPUS='"device=0"' \
  CONFIG_PATH=acestep-v15-xl-turbo LM_MODEL=acestep-5Hz-lm-4B \
  DURATIONS="30 60 120 240 480 600" STEPS="8" BATCHES="2" \
  bash /root/stress_acestep.sh
```
> 前置:host 装 ffmpeg(`apt-get install -y ffmpeg`)启用产物时长/静音防呆;`docker rm -f` 清场;tmux 里跑长压测。

---

## 9. 一页速查

| 维度 | 结论 |
|---|---|
| 生产默认 | **xl-turbo(4B DiT)+ 5Hz-lm-4B**;备选 turbo(2B,-5G) |
| 单卡副本 | 1 副本/卡 → 4×A100 节点 4 副本(或留 1 卡=3) |
| 显存峰值 | turbo 21G / xl-turbo 26.5G(600s,均 <40G) |
| 时长上限 | 600s 官方上限内无瓶颈(显存不随时长涨,co-scheduling Shmem<200M) |
| 每实例吞吐 | ~6 请求/min(workers=1 串行);RTF 0.08–0.17,快 6–12× |
| turbo vs xl-turbo | XL 音质略优、+5G、~同速;CLAP 无显著差异(但测的是贴合度非保真) |
| 1.7B vs 4B LM | 4B +6G 显存;默认 4B(卡够) |
| 唱歌 | lyrics 填真实歌词即唱,vocal_language 定语言;`[inst]`=器乐 |
| 踩坑 | 锁 cu128(驱动570);nano-vllm/flash_attn2 arm 可用;CLAP 打分需覆盖 baked 的 HF_HUB_OFFLINE=1 |
