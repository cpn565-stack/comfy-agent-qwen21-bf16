# Qwen Image 2.1：A100 非量化 BF16

**狀態：Starter。已完成本機靜態／CLI 檢查，尚未在 Colab A100 完成部署與出圖驗證。**

這是 `qwen_image_2_1` 套件的獨立 BF16 版本。生成器、Qwen3-VL 8B
文字編碼器和 VAE 都下載 BF16 權重；啟動時明確使用
`--bf16-unet --bf16-text-enc --bf16-vae`。編輯工作流的 KV cache 使用
`dtype: default`（無損）與 CPU 儲存，沒有 INT8／INT4 cache。

## 部署前條件

- 實際 GPU 必須是 **A100 40GB 或 80GB**，且 PyTorch 回報 CUDA 和 BF16 可用。
  腳本會在安裝／下載權重前檢查；T4、L4 和小容量 MIG 分割會直接停止。
- 此套件採保守的 **系統 RAM 至少 48 GiB** 門檻，建議使用高記憶體
  runtime。這是本套件的前置政策，並非原廠宣稱的最低需求。
- 冷啟動需要約 **40.21 GiB 可用磁碟**：30.21 GiB 權重加 10 GiB
  安裝與快取空間。重跑時只計算尚未完整下載的權重。
- A100 40GB 不保證所有解析度／多參考圖都能運行。先測 **1024×1024、
  batch 1**；ComfyUI 的正常記憶體管理可把組件移到 RAM，這不會量化權重。

## 固定來源

核對日期：2026-10-02。

- ComfyUI commit：`1b883beab11c04a2eb82cf6a3ee64294f3303897`。
  已確認包含 `TextEncodeQwenImage21`、`QwenImage21Cache` 與三個 BF16 CLI 選項。
- 模型來源：[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)。
- HF revision：`cb504a4090723e43f17ad01cec0359490e2de613`。
- 保留既有 Colab PyTorch／CUDA，以 pip constraints 安裝其他依賴。
  版本衝突會停止；不再使用原套件硬寫的 CUDA 12.1 安裝來源。
- 沒有額外的量化 prompt-enhancement 模型；使用原本輸入的 prompt。

| 組件 | 檔名 | bytes |
|---|---|---:|
| 生成器 | `diffusion_models/qwen_image_2.1_bf16.safetensors` | 14,230,280,616 |
| 文字編碼器 | `text_encoders/qwen3vl_8b_bf16.safetensors` | 17,534,334,616 |
| VAE | `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 |
| 合計 | | **32,440,124,920（32.44 GB／30.21 GiB）** |

這個合計是下載量，不是推論顯存峰值。下載後會核對檔案大小與 safetensors
header；必須含 BF16 tensor，並拒絕 INT8／FP8 tensor。少量 FP32 buffer 可保留。
程式碼沿用 MIT；模型沿用上游 Qwen Research License。

## 使用本 fork

在本機 clone 這個 fork 的 BF16 版本，然後執行：

```bash
npm ci
npm run build
node dist/cli/index.js colab kit qwen_image_2_1_bf16 --json
```

這會列出本機 setup、專用 launcher 與三個工作流的路徑。
若使用 npm registry 的上游 `comfy-agent`，它不會包含本 fork 的新增套件。
要讓 `comfy-agent` 指令使用本 fork 並安裝內附的 Codex skill，可執行：

```bash
npm link
comfy-agent skill install --agent codex
```

## 在已選定的 A100 部署

先用現有 Colab CLI 確認 session；此套件不會替你建立或切換 GPU：

```bash
colab --auth oauth2 status -s gpu-test
```

將 **本套件的** `01_setup.py` 和 `02_start_comfyui.py` 依序貼進兩個 Colab
儲存格執行，或從本機在 fork 根目錄執行：

```bash
colab --auth oauth2 exec -s gpu-test -f scripts/colab/qwen_image_2_1_bf16/01_setup.py
colab --auth oauth2 exec -s gpu-test -f scripts/colab/qwen_image_2_1_bf16/02_start_comfyui.py
colab --auth oauth2 exec -s gpu-test 'print(open("/content/qwen21_bf16_url.txt").read())'
```

等待 URL 檔出現後再讀取；下載需要時間。setup 可在遠端 SSH 用普通 Python
執行，但 **launcher 請在 Colab/Jupyter kernel 執行**：背景 tunnel thread
依附 kernel。不要在 SSH 用 `python 02_start_comfyui.py`，也不要重啟 kernel。

遠端 workspace：`/content/ComfyUI-Qwen21-BF16`。兩個腳本都可用環境變數
`QWEN21_BF16_WORKSPACE` 覆寫；launcher port 可用 `QWEN21_BF16_PORT` 覆寫。
專用 launcher 是必要的，它會檢查 A100 與 BF16 檔案並傳入 BF16 選項。

## 在本機連線與生成

```bash
node dist/cli/index.js connect https://YOUR-ID.trycloudflare.com
node dist/cli/index.js doctor --json
node dist/cli/index.js import scripts/colab/qwen_image_2_1_bf16/qwen_image_2_1_bf16_t2i.json --name qwen21_bf16_t2i
node dist/cli/index.js run qwen21_bf16_t2i --prompt 'A cinematic photograph of a rainy Taipei street at night' --seed 42
```

另外兩個工作流是 `qwen_image_2_1_bf16_edit.json`（單圖編輯）與
`qwen_image_2_1_bf16_edit_2ref.json`（原圖＋參考圖），匯入後可用 `--image`
和 `--image-2` 上傳。預設 1024 像素預算、batch 1、40 steps、cfg 1。

生成結果會由本機 CLI 下載到 `.comfy-agent/outputs/`。Colab URL 隨新 runtime
改變；重新 `connect` 即可。遠端 `/content` 隨 runtime 刪除。

## 驗證記錄

本版本目前僅通過本機檢查，不承接上游 **INT8 L4** 的 Verified 標記。
要升為 Verified，必須在一次 A100 session 完成：setup、launcher／tunnel、
本機 doctor、工作流 import、run 及本機收到圖片，並實際檢視輸出。

GPU 推論時間、2K 最大顯存、單圖／雙參考編輯的實際可用性都尚待實測。

參考：[ComfyUI 官方 Qwen Image 2.1 文件](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)。
