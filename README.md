# GraphRAG 开发说明

## 项目定位

这是一个面向文档问答的多模态检索增强生成（RAG）原型。主流程支持将 PDF 解析为 Markdown，保留文档中的图文信息，通过 Milvus 检索相关上下文，并由 LangGraph 编排多模态问答流程。

## 主功能范围

项目主功能包括：

- PDF/OCR 解析
- Markdown 文档处理
- Milvus 向量检索
- LangGraph 工作流
- 多模态问答
- Gradio 与 Vue 前端

> Neo4j 相关文件是独立的练习/实验代码，不属于主功能，也不是运行本项目的前置依赖。

## 目录结构

```text
.
├── mian.py                 # PDF 上传与 OCR 解析的 Gradio 界面入口
├── dots_ocr/               # PDF/图片解析、版面识别与 Markdown 输出
├── splister/               # Markdown 切分、标题层级整理和图片处理
├── milvus_db/              # Milvus 集合定义与检索器
├── workflow/               # LangGraph 问答工作流、状态、路由与工具
├── utils/                  # 配置、日志、图片编码和通用辅助函数
├── llm.py                  # 大模型客户端封装
├── embeding.py              # 文本嵌入模型封装
├── font/                   # Vue 3 + Vite 前端
├── output/                 # OCR 解析生成的 Markdown、JSON 和图片
├── readme.txt              # Python 依赖清单
└── connect_data.py         # Neo4j 实验脚本（不属于主流程）
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `mian.py` | 提供 PDF 上传、调用 OCR 解析，并展示生成的 Markdown 内容。 |
| `dots_ocr/` | 将 PDF 或图片转为页面图像，执行版面/内容解析，输出 Markdown、JSON 和图片资源。 |
| `splister/splister_md.py` | 按 Markdown 标题和语义切分文档，提取图片并整理为可入库的文档片段。 |
| `embeding.py` | 通过 `CustomQwen3Embeddings` 生成文本向量。 |
| `milvus_db/collections_operator.py` | 定义文档与对话上下文集合，包括 BM25 稀疏向量和稠密向量字段。 |
| `milvus_db/db_retriever.py` | 提供文本、图像及混合检索能力，返回正文、来源文件和关联图片路径。 |
| `workflow/Workflow.py` | 构建 Gradio 问答界面，组装 LangGraph 节点、检查点与执行流程。 |
| `workflow/search_node.py`、`workflow/tools.py` | 执行知识库检索、组织检索结果，并提供工具调用接口。 |
| `workflow/state.py`、`workflow/alll_route.py` | 定义工作流状态与条件路由规则。 |
| `workflow/save_context.py` | 将问答结果异步写入 Milvus 上下文集合。 |
| `font/` | Vue 前端页面和请求封装；后端 API 需与 `font/src/api/chat.js` 中的接口约定保持一致。 |

## 主工作流

```text
PDF/图片
  → dots_ocr 解析
  → Markdown、页面 JSON 与图片资源
  → splister 切分并保留标题/图片上下文
  → 向量化并写入 Milvus

文本或图片提问
  → workflow/Workflow.py 接收输入
  → LangGraph 判断输入类型与检索路径
  → Milvus 返回相关文本和图片
  → 大模型依据检索上下文生成回答
  → 可选评估、人工确认与网络检索分支
  → 问答结果写入 Milvus 上下文集合
```

`Workflow.py` 中的核心节点包括输入预处理、工具调用、检索、答案生成、答案评估、人工确认和网络检索。状态字段由 `MultiModalRAGState` 统一维护，条件分支由 `alll_route.py` 决定。

## 开发注意事项

- Python 依赖记录在 `readme.txt`；前端依赖与命令位于 `font/package.json`。
- 项目依赖外部服务：Milvus、大模型/嵌入服务，以及工作流检查点和长期记忆所用的 PostgreSQL。运行前应将服务地址和密钥移入环境变量或本地配置，避免提交真实凭据。
- `utils/env_utils.py` 和 `util.py` 会从环境变量读取模型服务配置。建议提供本地 `.env` 文件，并将其加入 `.gitignore`。
- 文档集合的稠密向量维度定义为 1024；替换嵌入模型时，需同步检查集合 schema、索引和检索逻辑。
- OCR 输出默认落在 `output/`。该目录可能快速增长，建议将样例输出与运行时产物分开管理。
- `mian.py` 是 PDF 解析界面入口，`workflow/Workflow.py` 是 Gradio 问答入口；Vue 前端的 API 路由需由独立后端实现或代理提供。
- 代码中存在实验性分支和不同版本的工作流实现（如 `Workflow1.py`）。修改主流程前，先确认实际使用的入口和工作流文件。
- 不要将 `connect_data.py`、`write_to_neo4j.py` 等 Neo4j 练习脚本纳入部署、依赖安装或主流程说明。

