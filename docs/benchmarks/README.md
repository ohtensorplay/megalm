# 基准

## 记录要求

性能数字必须同时给出：仓库 commit、MegaLM 源码摘要、驱动 / CUDA / Torch 版本、模型 artifact、dtype / 量化、请求集、并发、输出长度、特性开关、硬件。缺项的数字只能用于定位，不能作为结论。

对比基线时还要保证同模型、同精度、同请求集、同功能开关。关闭基线特性换来的领先不算结果。每项优化需要独立消融。

## 结果

| 文档 | 内容 |
| --- | --- |
| [vs vLLM](oxr-vllm.md) | OXR-1.0 视觉语言模型，单卡 RTX 3090，四组服务场景 |
| [vs llama.cpp](llamacpp.md) | 同 GPU、同 GGUF、同启动参数，18 个吞吐 / 延迟点 |
| [vs SGLang](qwen3vl-awq-sglang.md) | Qwen3-VL-8B AWQ int4，单卡 RTX 3090，b1/b8/b16，含块大小与内核消融 |
| [LoRA](lora-benchmark.json) | 4 层合成模型上启用 / 禁用 LoRA 的中位延迟 |

图表源文件：`oxr_vllm_comparison.png`、`oxr_vllm_comparison.pdf`、`llamacpp_comparison.png`、`llamacpp_prefix_hit.png`。`make_oxr_comparison_chart.py` 从原始 JSON 生成 vLLM 对比图。
