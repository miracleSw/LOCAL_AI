# LOCAL_AI_CORE — Tài liệu Kỹ thuật & Hướng dẫn Vận hành

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Inference Engine](https://img.shields.io/badge/Engine-llama.cpp%20(Vulkan)-red.svg)](https://github.com/ggerganov/llama.cpp)
[![Default Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--3B--Q4__K__M-green.svg)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English Version](README.md) | **Tiếng Việt**

Khung điều phối quy trình (Pipeline Framework) chạy mô hình ngôn ngữ 3B (Qwen 2.5 Coder 3B GGUF) hoàn toàn cục bộ (offline) trên phần cứng tối thiểu (2 GB VRAM / 8 GB RAM).

Dự án loại bỏ hoàn toàn các thư viện phụ thuộc cồng kềnh bằng cách tích hợp trực tiếp: bộ tìm kiếm BM25 thuần Python, quản lý trạng thái phiên làm việc trong RAM, cơ chế thực thi kép (HTTP Server + CLI Fallback) và vòng lặp tự sửa lỗi cú pháp.

---

## 1. Điểm Khác Biệt & Quyết Định Kỹ Thuật

* **Không phụ thuộc thư viện nặng (Zero Heavyweight Dependencies):**
  Thuật toán BM25, tách từ ngữ, chuẩn hóa Unicode NFD và bóc tách tài liệu đều sử dụng 100% thư viện chuẩn của Python (`math`, `re`, `unicodedata`). Không cần cài đặt ChromaDB, FAISS hay PyTorch, tiết kiệm ngay 1.5–2.5 GB RAM khi khởi động.
* **Cơ chế thực thi kép (Triệt tiêu độ trễ nạp lại mô hình):**
  * *Chế độ chính:* Kết nối qua HTTP tới tiến trình nền `llama-server.exe` (cổng 8080). Trọng số mô hình 2.1 GB được giữ nguyên trên RAM/VRAM, giảm thời gian nạp lại mô hình giữa các bước về 0 giây và tái sử dụng KV-cache.
  * *Chế độ dự phòng:* Tự động chuyển sang gọi `llama-cli.exe` nếu server chưa được bật.
* **Cô lập trạng thái trong bộ nhớ RAM (Chống rò rỉ dữ liệu cũ):**
  Toàn bộ biến, ngữ cảnh và dữ liệu giữa các bước được quản lý thông qua đối tượng `PipelineSession` trong RAM. Không còn tình trạng các bước sau đọc nhầm file markdown cũ của đề thi trước trên ổ đĩa.
* **Thẩm định cú pháp & Tự sửa lỗi (Self-Repair Loop):**
  Mỗi bước sinh mã đều đi qua validator. Nếu phát hiện vi phạm cú pháp, hệ thống sẽ trích xuất mã lỗi và vị trí vi phạm để gửi lại prompt sửa lỗi có chủ đích, không làm phình context window.
* **Tăng tốc phần cứng đa nền tảng qua Vulkan:**
  Sử dụng prebuilt `llama.cpp` Vulkan (`ggml-vulkan.dll`), tận dụng VRAM của card đồ họa NVIDIA, AMD Radeon (RDNA) và Intel (Arc/Iris/iGPU) trên Windows mà không cần cài đặt CUDA SDK.

---

## 2. Sơ đồ Kiến trúc Luồng Dữ liệu

```
                                    +-----------------------+
                                    | input/current_task.txt|
                                    +-----------+-----------+
                                                |
                                                v
+------------------------+          +-----------+-----------+          +-------------------------+
|      rag/active/       |          |      Orchestrator     |          |     PipelineSession     |
| (Quy tắc & Code mẫu)   | -------> |  (core/orchestrator)  | <------> |  (Quản lý trạng thái &  |
|   [Pure BM25 Engine]   |          +-----------+-----------+          |   Dọn dẹp bộ nhớ RAM)   |
+------------------------+                      |                      +-------------------------+
                                                v
                                    +-----------+-----------+
                                    |       LLMClient       |
                                    |    (core/llm_client)  |
                                    +-----+-----------+-----+
                                          |           |
                           [Server Online]|           |[Fallback Offline]
                                          v           v
                             +----------------+   +---------------+
                             |  llama-server  |   |   llama-cli   |
                             | (Cổng 8080)    |   | (Chạy độc lập)|
                             +----------------+   +---------------+
                                          \           /
                                           v         v
                                    +-----------------------+
                                    |    BaseValidator      |
                                    | (Kiểm tra cú pháp)    |
                                    +-----------+-----------+
                                          |           |
                                    [Lỗi] |           | [Hợp lệ]
                                          v           v
                                    +-----------+ +------------------------+
                                    |Vòng lặp   | |  output/final_project/ |
                                    |tự sửa lỗi | |  (Mã nguồn hoàn chỉnh) |
                                    +-----------+ +------------------------+
```

---

## 3. Cân Đối Tài Nguyên & Tối Ưu Phần Cứng

Cấu hình mặc định được tinh chỉnh chuyên sâu cho máy tính có cấu hình văn phòng / phòng thi (kiểm nghiệm trên Intel Core i7-12700, 8 GB RAM, AMD Radeon RX 6300 2 GB GDDR6):

### 1. Phân bổ VRAM (2048 MB tổng)
Cấu hình tối ưu đo đạc thực nghiệm trên hệ thống (AMD Radeon RX 6300 2GB VRAM):

| Thành phần | Dung lượng | Giải thích kỹ thuật |
| :--- | :--- | :--- |
| **Trọng số 20 layer GPU** | ~1,140 MB | Đưa 20 / 36 block transformer lên VRAM qua Vulkan (tối ưu tối đa trước ngưỡng trào PCIe). |
| **Vulkan Compute Scratch Buffer** | ~305 MB | Vùng nhớ đệm thực thi đồ thị tính toán ma trận với micro-batch 512. |
| **KV Cache 20 layer (@ 6144 ctx)**| ~120 MB | Rất nhẹ nhờ cơ chế GQA (tỷ lệ 8:1, 2 KV heads) của Qwen 2.5. |
| **Windows DWM + Desktop Display** | ~350–400 MB | Bộ nhớ giao diện desktop của hệ điều hành. |
| **Tổng VRAM sử dụng** | **~1,915 MB** | **Đạt ngưỡng tối ưu 93% VRAM**, sinh token đạt ~17.5 - 18.2 tok/s. |

> **Cảnh báo kỹ thuật:** Ngưỡng sụt giảm hiệu năng (Performance Cliff) nằm ở `gpu_layers >= 23`. Khi vượt quá 22 layer, VRAM bị tràn và Windows WDDM buộc phải trào dữ liệu sang RAM hệ thống qua khe cắm PCIe 4.0 x4 (băng thông tụt từ 64 GB/s xuống dưới 8 GB/s), khiến tốc độ sinh token tụt nghiêm trọng từ **18.3 t/s xuống 11.1 t/s**. Do đó, `gpu_layers = 20` là điểm ngọt an toàn tuyệt đối, vừa đạt đỉnh tốc độ vừa không làm crash máy.

### 2. Tối ưu Flash Attention trên Vulkan Shader
* Khi để `flash_attn = auto` (bật): Tốc độ xử lý prompt dài (2048 tokens) chỉ đạt **143.42 tok/s**.
* Khi tắt Flash Attention (`--flash-attn off`): Tốc độ xử lý prompt nhảy vọt lên **333.90 – 360 tok/s** (tăng tốc **2.32 lần** / +133%). Lý do: Kernel attention tiêu chuẩn trên tập lệnh Vulkan chạy song song hiệu quả hơn nhiều so với kernel flash-attention thử nghiệm trên card AMD RDNA2.

### 3. Phân bổ RAM hệ thống (8192 MB tổng)

| Thành phần | Dung lượng |
| :--- | :--- |
| **Hệ điều hành Windows + Dịch vụ nền** | ~3,400–3,800 MB |
| **16 layer mô hình trên CPU** | ~860 MB |
| **KV Cache 16 layer trên CPU (@ 6144 ctx)** | ~96 MB |
| **Python Runtime + Chỉ mục BM25** | ~80 MB |
| **Tổng RAM tiêu thụ** | **~4,650 MB** (~3.4 GB trống an toàn cho các tác vụ khác) |

### 4. Tối ưu số luồng CPU trên vi kiến trúc lai (Hybrid CPU)
Với CPU có cả nhân P-core và E-core (như i7-12700 gồm 8 P-cores và 4 E-cores), cấu hình tối ưu là `threads = 8`. GGML sử dụng cơ chế rào cản đồng bộ (Synchronous Barrier) trên mỗi layer tính toán; nếu đặt `threads > 8`, luồng tính toán sẽ bị đẩy sang các nhân E-core có xung nhịp và IPC thấp hơn, khiến 8 nhân P-core mạnh phải dừng chờ nhân E-core tại điểm đồng bộ.

---

## 4. Cấu Trúc Thư Mục

```text
LOCAL_AI_CORE/
├── config.ini                  # Cấu hình tập trung (Server port, context, luồng, GPU layer)
├── 00_MENU.bat                 # Menu điều khiển giao diện dòng lệnh tương tác UTF-8
├── run_server.bat              # Script khởi động persistent Vulkan llama-server
├── stop_server.bat             # Script dừng llama-server an toàn
├── README.md                   # Tài liệu tiếng Anh chuẩn GitHub
├── README_VI.md                # Tài liệu kỹ thuật tiếng Việt
│
├── core/                       # Thư viện lõi độc lập
│   ├── config.py               # Loader cấu hình an toàn, kiểm tra đường dẫn
│   ├── bm25_rag.py             # Bộ tìm kiếm BM25 thuần Python (tách từ Việt/Anh)
│   ├── llm_client.py           # Dual-mode HTTP server + CLI fallback client
│   ├── pipeline_session.py     # Quản lý trạng thái in-memory & snapshot JSON
│   ├── validator_base.py       # Framework kiểm tra cú pháp & vòng lặp tự sửa lỗi
│   ├── exporter.py             # Trích xuất mã nguồn và xuất ra thư mục sản phẩm
│   └── orchestrator.py         # CLI entrypoint (--run-all, --step, --check, --status, --reset)
│
├── steps/                      # Định nghĩa các bước xử lý bài toán mẫu
│   ├── step_01_analyze.py      # Bóc tách yêu cầu, trích xuất ràng buộc & ca biên
│   ├── step_02_solve.py        # Sinh kiến trúc giải pháp và toàn bộ mã nguồn thực thi
│   └── step_03_verify.py       # Rà soát ràng buộc, kiểm định chất lượng & đóng gói
│
├── models/
│   └── qwen25-coder-3b-q4km.gguf   # Model Qwen 2.5 Coder 3B GGUF
│
├── llama-vulkan/               # Bộ binary llama.cpp chạy Vulkan offline (52 files)
│   ├── llama-server.exe
│   ├── llama-cli.exe
│   └── ggml-vulkan.dll
│
├── rag/active/                 # Tài liệu RAG tri thức nghiệp vụ
│   ├── 00_SYSTEM_RULES.md
│   └── 01_PROBLEM_SOLVING_FRAMEWORK.md
│
├── input/
│   └── current_task.txt        # Đề bài hoặc yêu cầu bài toán cần giải quyết
│
└── output/
    ├── steps/                  # Nhật ký markdown của từng bước
    ├── final_project/          # Mã nguồn và file thực thi hoàn chỉnh xuất ra
    └── logs/                   # Báo cáo kiểm định và snapshot session
```

---

## 5. Hướng Dẫn Sử Dụng

### Yêu cầu môi trường
* Windows 10 / 11 (64-bit).
* Python 3.10 trở lên (đã tích hợp vào biến môi trường `PATH`).
* Driver card đồ họa hỗ trợ Vulkan (NVIDIA, AMD hoặc Intel).

### Cách 1: Sử dụng Menu tương tác (Khuyến nghị)
1. Chạy file **`00_MENU.bat`**.
2. Chọn **`[1]`** để khởi động **Llama Server** (cửa sổ server chạy ngầm trên cổng 8080).
3. Chọn **`[3]`** để dán đề bài hoặc yêu cầu bài toán vào file `input\current_task.txt`.
4. Chọn **`[6]`** (**RUN ALL**) để hệ thống tự động chạy toàn bộ quy trình.
5. Khi hoàn tất, chọn **`[13]`** để mở thư mục `output\final_project` chứa toàn bộ code hoàn chỉnh.

### Cách 2: Sử dụng dòng lệnh (Dành cho Developer / CI)

```powershell
cd d:\Project\Model_Local_Test\LOCAL_AI_CORE

# 1. Kiểm tra tính toàn vẹn hệ thống
python core/orchestrator.py --check

# 2. Xem trạng thái hiện tại của pipeline
python core/orchestrator.py --status

# 3. Chạy toàn bộ quy trình tự động từ đầu đến cuối
python core/orchestrator.py --run-all

# 4. Hoặc truyền đề bài trực tiếp qua dòng lệnh
python core/orchestrator.py --task "Viết class ThreadPoolExecutor tối giản bằng Python kèm unit test" --run-all

# 5. Chạy riêng lẻ từng bước
python core/orchestrator.py --step analyze
python core/orchestrator.py --step solve
python core/orchestrator.py --step verify

# 6. Xuất mã nguồn ra output/final_project
python core/orchestrator.py --export

# 7. Dọn dẹp sạch sẽ phiên làm việc cũ
python core/orchestrator.py --reset
```

---

## 6. Tham Số Cấu Hình (`config.ini`)

```ini
[paths]
llama_server = llama-vulkan/llama-server.exe
llama_cli    = llama-vulkan/llama-cli.exe
model        = models/qwen25-coder-3b-q4km.gguf
rag_dir      = rag/active
input_task   = input/current_task.txt
output_dir   = output

[server]
host                   = 127.0.0.1
port                   = 8080
endpoint               = http://127.0.0.1:8080/completion
request_timeout_seconds= 300

[llama]
ctx_size            = 6144      # Context size in tokens (bội số của 1024)
threads             = 8         # 8 luồng khớp 8 nhân P-core của i7-12700
gpu_layers          = 20        # 20 layer tối ưu đỉnh cao cho VRAM 2GB (RX 6300)
no_mmap             = true      # Khóa trang nhớ trong RAM, tránh đơ do page fault SSD
no_kv_offload       = false     # Cho phép GPU giữ KV cache 20 layer để tăng tốc sinh mã
flash_attn          = false     # Tắt Flash Attention để tăng tốc 2.3x prompt processing trên Vulkan
temperature         = 0.05      # Nhiệt độ thấp cho code mang tính xác định
top_p               = 0.85
repeat_penalty      = 1.05
max_tokens          = 2048
max_runtime_seconds = 300

[rag]
top_k               = 5         # Số chunk BM25 trích xuất mỗi bước
chunk_chars         = 1200      # Độ dài ký tự mỗi đoạn chunk
overlap_chars       = 150       # Độ gối đầu giữa các đoạn liền kề
always_include_rules= true      # Luôn nạp các tài liệu có tiền tố '00_'

[steps]
step_01_max_tokens  = 1200
step_02_max_tokens  = 2048
step_03_max_tokens  = 2048
max_repair_attempts = 2         # Số lần thử tự sửa lỗi tối đa
```

---

## 7. Tùy Biến Nghiệp Vụ Riêng (Custom Domain Adaptation)

Hệ thống được thiết kế tách rời khỏi nghiệp vụ cụ thể. Để áp dụng cho các bài toán khác (như sinh mã SQL, thiết kế Java OOP, viết API):

1. **Bổ sung tài liệu RAG:** Thả các file markdown (`.md`), tài liệu chuẩn hoặc code mẫu vào `rag/active/`. Đặt tiền tố `00_` cho các quy tắc bắt buộc model luôn phải tuân thủ (ví dụ: `00_SQL_SERVER_2014_RULES.md`).
2. **Viết Step mới:** Tạo hàm xử lý trong `steps/` kế thừa từ `core.pipeline_session.PipelineSession` và `core.llm_client.LLMClient`.
3. **Thêm bộ kiểm tra:** Kế thừa lớp `BaseValidator` trong `core/validator_base.py` để định nghĩa các điều kiện kiểm duyệt cú pháp hoặc từ khóa cấm.

---

## 8. Bộ Kiểm Thử (Regression Test Suite)

Dự án tích hợp sẵn bộ unit test tự động bằng thư viện `unittest` chuẩn của Python:

```powershell
python -m unittest discover -s tests
```

Bao gồm các bài kiểm tra:
* Kiểm tra tải cấu hình strongly-typed và phân giải đường dẫn tuyệt đối.
* Kiểm tra chỉ mục BM25, tách từ ngữ tiếng Việt/tiếng Anh và chuẩn hóa bỏ dấu.
* Kiểm tra cô lập phiên làm việc trong RAM và cơ chế dọn dẹp atomic.
* Kiểm tra format chẩn đoán lỗi của validator và prompt tự sửa lỗi.
* Kiểm tra an toàn chống path traversal khi trích xuất file mã nguồn.

---

## Bản Quyền (License)

Dự án được phát hành theo giấy phép [MIT License](LICENSE). Mã nguồn được thiết kế phục vụ môi trường Offline AI tốc độ cao, ổn định và bảo mật dữ liệu.
