# 模型和数据

仓库不分发模型权重、评测 TSV 或逐样本输出。

- Qwen 模型官方入口：[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)。下载前核对模型条款及 revision；原比赛应使用主办方指定的固定版本。
- MMBench 官方项目：[open-compass/MMBench](https://github.com/open-compass/MMBench)。按照官方说明取得公开英文/中文开发集，遵守对应数据使用条款。

本项目入口期望以下路径及带图片 base64 的 TSV 字段：

```text
Qwen3.5-2B/                 模型 config、tokenizer、processor 和权重
datasets/mmbench/
  mmbench_dev_en.tsv
  mmbench_dev_cn.tsv
```

TSV 使用 `index, question, hint, A, B, C, D, answer, image, category, l2-category` 等字段。原包中的数据文件超过 120 MB，各自不应直接加入普通 Git 历史。避免仅为绕过大小限制而随意换名分块发布未经确认可再分发的数据。
