# 06_deepseek_setup.ps1 -- 一键配置 DeepSeek 后端（VADAR 对照臂）并自检
#
# !!! 本文件必须以 UTF-8 **带 BOM** 保存 !!!
# ---------------------------------------------------------------------------
# 原因：Windows PowerShell 5.1 读 .ps1 时，如果文件没有 BOM，会按系统 ANSI 代码页
# （中文机 = GBK）解码。于是本文件的中文注释被解成乱码，而乱码字节里可能出现
# **游离的引号**，直接导致 ParserError —— 报错行号还会指向很远的另一行，极难排查。
#   - 本目录其余 .ps1 都是**纯 ASCII**（0 个非 ASCII 字节），所以它们没这个问题；
#   - 本文件为了保留中文告警而带 BOM。用编辑器另存时**务必保留 BOM**，
#     否则脚本会在打印任何东西之前就解析失败。
# 校验：读前 3 字节应为 EF BB BF。
# ---------------------------------------------------------------------------
#
# 为什么需要这个脚本
# ------------------
# VADAR 原版把 gpt-4o 和 ./api.key 硬编码在源码里（engine_utils.py:56-57），
# 而本机 api.openai.com 网络层不可达，所以必须换后端。03_vadar_llm_bridge.py 用
# 环境变量驱动的适配器做零侵入替换 —— 但环境变量有十几个，且**其中一个设错会
# 静默产生错误实验结论**（见下方「思考模式」告警）。集中在这里设一次，避免手输漏项。
#
# 用法（在 phase0 目录下）
# ------------------------
#     $env:DEEPSEEK_API_KEY = "sk-xxxx"
#     .\06_deepseek_setup.ps1                 # 关思考 + 温度 0.2（可复现优先，默认）
#     .\06_deepseek_setup.ps1 -KeepThinking   # 留思考（质量优先，但温度无效）
#     .\06_deepseek_setup.ps1 -ShowOnly       # 只打印配置，不发任何请求
#     .\06_deepseek_setup.ps1 -SkipProbe      # 设好环境变量后直接进交互式会话
#     .\06_deepseek_setup.ps1 -PriceIn 1 -PriceOut 2   # 带单价，报告里给出费用外推
#
# 注意：环境变量只在本会话有效。要跑实验就在**同一个窗口**里接着跑。

param(
    [string]$ApiKey = $env:DEEPSEEK_API_KEY,
    [string]$BaseUrl = "https://api.deepseek.com",
    [string]$Model = "deepseek-flash",
    [int]$MaxTokens = 8192,
    [double]$Temperature = 0.2,
    [double]$PriceIn = 0.0,
    [double]$PriceOut = 0.0,
    [switch]$KeepThinking,
    [switch]$ShowOnly,
    [switch]$SkipProbe
)

$ErrorActionPreference = "Stop"

# 让中文与 emoji 不乱码（控制台默认是 GBK）。
# ⚠ 必须 try/catch：当 stdout 被重定向（> file）时 [Console] 句柄无效，
#   给 OutputEncoding 赋值会抛 "The handle is invalid"。若不加 try，
#   配合上面的 ErrorActionPreference="Stop"，脚本会**在打印任何东西之前就死掉**，
#   重定向文件里只剩空白 —— 极难排查。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---- 选 python：优先项目 venv，退回 PATH ----
$py = Join-Path (Split-Path -Parent $here) "venvs\vision\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$repo = Split-Path -Parent $here

# ---- 1. 校验 key ----
if (-not $ShowOnly -and -not $ApiKey) {
    Write-Host "缺少 API key。" -ForegroundColor Red
    Write-Host "  先设环境变量:  `$env:DEEPSEEK_API_KEY = 'sk-xxxx'"
    Write-Host "  或传参:        .\06_deepseek_setup.ps1 -ApiKey sk-xxxx"
    Write-Host "  key 在 https://platform.deepseek.com 的 API keys 页面创建。"
    exit 2
}

# ---- 2. 环境变量（这是 03_vadar_llm_bridge.py 读的全部入参）----
$env:VADAR_BASE_URL    = $BaseUrl
$env:VADAR_API_KEY     = $ApiKey
$env:VADAR_MODEL       = $Model
$env:VADAR_TEMPERATURE = "$Temperature"
$env:VADAR_MAX_TOKENS  = "$MaxTokens"
$env:VADAR_MAX_RETRIES = "2"
$env:VADAR_REPO        = Join-Path $repo "vendor\VADAR"
$env:VADAR_CALL_LOG    = Join-Path $repo "logs\vadar_llm_calls.jsonl"

# 视觉端点不单独设 -> bridge 里 vision = text，要求主模型自己多模态。
# 主模型只要选对了（vision-exp），vqa() 就能跑。想省钱再配 VADAR_VISION_* 走 Qwen3-VL。
Remove-Item Env:\VADAR_VISION_BASE_URL -ErrorAction SilentlyContinue
Remove-Item Env:\VADAR_VISION_MODEL    -ErrorAction SilentlyContinue
Remove-Item Env:\VADAR_VISION_API_KEY  -ErrorAction SilentlyContinue

# ---- 3. 思考模式：这里必须显式表态 ----
if ($KeepThinking) {
    Remove-Item Env:\VADAR_EXTRA_BODY -ErrorAction SilentlyContinue
    $thinking = "开（DeepSeek 默认，effort=high）"
} else {
    $env:VADAR_EXTRA_BODY = '{"thinking": {"type": "disabled"}}'
    $thinking = "关（extra_body 显式 disabled）"
}

# ---- 4. 打印将要生效的配置 ----
Write-Host ""
Write-Host "=================================================================="
Write-Host " DeepSeek 后端配置" -ForegroundColor Cyan
Write-Host "=================================================================="
Write-Host "  base_url    : $BaseUrl"
Write-Host "  model       : $Model"
if ($ApiKey) {
    Write-Host "  api_key     : ***$($ApiKey.Substring([Math]::Max(0, $ApiKey.Length - 4)))"
} else {
    Write-Host "  api_key     : (未设)"
}
Write-Host "  temperature : $Temperature"
Write-Host "  max_tokens  : $MaxTokens"
Write-Host "  思考模式    : $thinking"
$ebShown = if ($env:VADAR_EXTRA_BODY) { $env:VADAR_EXTRA_BODY } else { "(未设 -> 端点默认)" }
Write-Host "  extra_body  : $ebShown"
Write-Host "  VADAR_REPO  : $env:VADAR_REPO"
Write-Host ""
Write-Host " [!] DeepSeek 思考模式下 temperature/top_p 等参数**无效**" -ForegroundColor Yellow
Write-Host "     官方文档：设置不报错，但也不会有任何效果。" -ForegroundColor Yellow
if ($KeepThinking) {
    Write-Host "     当前留了思考 -> temperature=$Temperature 是空操作，" -ForegroundColor Yellow
    Write-Host "     报告里不要再声称「低温可复现」，改用 n>1 重复采样 + 报告方差。" -ForegroundColor Yellow
} else {
    Write-Host "     当前关了思考 -> temperature=$Temperature 真正生效。" -ForegroundColor Green
}
Write-Host "=================================================================="
Write-Host ""

if ($ShowOnly) {
    Write-Host "(-ShowOnly：仅打印配置，未发任何请求)"
    exit 0
}

# ---- 5. 自检一：适配器配置（不需要 VADAR / torch）----
Write-Host "[1/2] 适配器配置自检 ..." -ForegroundColor Cyan
& $py (Join-Path $here "03_vadar_llm_bridge.py") --show-config

if ($SkipProbe) {
    Write-Host ""
    Write-Host "(-SkipProbe：环境变量已就绪，可在本窗口继续跑实验)" -ForegroundColor Green
    exit 0
}

# ---- 6. 自检二：真实调用探针 ----
#   T1 连通/模型名/思考探测 -> T1b 思考模式 A/B -> T2/T3/T4 标签契约 -> T5 错误路径 -> T6 token 记账
Write-Host ""
Write-Host "[2/2] 真实调用探针（会消耗少量 token）..." -ForegroundColor Cyan
$report = Join-Path $here "llm_probe_report.json"
# [!] 这里**故意不传** --extra-body，改由探针自己从 VADAR_EXTRA_BODY 环境变量读。
#     两条命令行传法 2026-09-17 都实测炸过（真探针，exit=2）：
#       a) 默认路径：显式传参时，本宿主向原生 exe 传含双引号的字符串会
#          **剥掉内层双引号**，探针收到的是 {thinking: {type: disabled}}
#          -> 不是合法 JSON -> return 2，整轮自检直接失败。
#       b) -KeepThinking 路径：env 变量被删 -> 该变量为 $null ->
#          本宿主会**整个丢掉这个参数**，argv 变成 `--extra-body --price-in 0 ...`，
#          argparse 报 "argument --extra-body: expected one argument" -> 同样 return 2。
#     而探针 argparse 的默认值本来就是读环境变量 VADAR_EXTRA_BODY，
#     与 03_vadar_llm_bridge.py 读的是**同一个来源**，所以不传反而才真正「同源」：
#       env 有值 -> 探针发给端点的就是它；env 被删 -> 探针用端点默认（思考开启）。
#     权威证据 = 报告抬头那行 `extra_body : ...`（打印的是实际解析成功的内容）。
& $py (Join-Path $here "02_probe_llm_api.py") `
    --base-url $BaseUrl `
    --model $Model `
    --api-key $ApiKey `
    --temperature $Temperature `
    --max-tokens $MaxTokens `
    --price-in $PriceIn `
    --price-out $PriceOut `
    --report $report
Write-Host "      提示：费用外推填单价 —— .\06_deepseek_setup.ps1 -PriceIn <元/百万> -PriceOut <元/百万>" -ForegroundColor DarkGray
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "自检完成。报告: $report" -ForegroundColor Green
    Write-Host "下一步：把 t1-t5 的结论抄进 docs 的实验章节（每个数字都是本机实测）。"
} else {
    Write-Host "自检未通过 (exit=$code)。先看报告: $report" -ForegroundColor Red
    Write-Host "  401/403 -> key 无效或未开通"
    Write-Host "  400 model not found -> 模型名写错（旧名 deepseek-chat 已于 2026-07-24 退役）"
    Write-Host "  400 且含 image -> 把图发给了纯文本模型（本脚本默认 vision-exp，不该出现）"
}
exit $code