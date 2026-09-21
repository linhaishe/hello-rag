# 版本升级架构对比

## 总体流程

| 阶段 | 原项目实现 | 升级后实现 | 主要代码位置 |
| --- | --- | --- | --- |
| 用户界面 | Gradio 旧版 API | Gradio 新版 API | `serve/run_gradio.py` |
| API 服务 | FastAPI | FastAPI | `serve/api.py` |
| 文档加载 | `langchain.document_loaders` | `langchain_community.document_loaders` | `database/create_db.py` |
| PDF 解析 | `PyMuPDFLoader` | `PyMuPDFLoader` | `database/create_db.py` |
| Markdown/TXT 解析 | Unstructured Loader | Community Unstructured Loader | `database/create_db.py` |
| 文本切分 | `langchain.text_splitter` | `langchain_text_splitters` | `database/create_db.py` |
| Prompt | `langchain.prompts` | `langchain_core.prompts` | `qa_chain/` |
| 问答链 | `langchain.chains` | `langchain_classic.chains` | `qa_chain/` |
| 对话记忆 | `langchain.memory` | `langchain_classic.memory` | `qa_chain/Chat_QA_chain_self.py` |
| Chroma 向量库 | `langchain.vectorstores.Chroma` | `langchain_chroma.Chroma` | `database/create_db.py`、`qa_chain/` |
| OpenAI 模型 | `langchain.chat_models.ChatOpenAI` | `langchain_openai.ChatOpenAI` | `qa_chain/model_to_llm.py` |
| OpenAI Embedding | `langchain.embeddings.openai` | `langchain_openai` | `embedding/`、`qa_chain/` |
| HuggingFace Embedding | `langchain.embeddings.huggingface` | `langchain_huggingface` | `embedding/call_embedding.py` |
| 自定义 LLM 基类 | `langchain.llms.base.LLM` | `langchain_core.language_models.llms.LLM` | `llm/` |
| 回调类型 | `langchain.callbacks` | `langchain_core.callbacks` | `llm/` |
| Pydantic 兼容层 | `langchain.pydantic_v1` | `pydantic.v1` | `embedding/`、`llm/` |
| LangChain 工具函数 | `langchain.utils` | `langchain_core.utils` | `embedding/`、`llm/` |

## 依赖架构

| 功能 | 原依赖 | 升级后依赖 |
| --- | --- | --- |
| LangChain 核心 | `langchain` 单体包 | `langchain`、`langchain-core` |
| 旧版 Chain | 内置于 `langchain` | `langchain-classic` |
| Chroma 集成 | `chromadb==0.3.29` | `chromadb>=1.0,<2`、`langchain-chroma` |
| OpenAI 集成 | `langchain` 内置 | `langchain-openai` |
| HuggingFace 集成 | `langchain` 内置 | `langchain-huggingface` |
| Community 组件 | `langchain` 内置 | `langchain-community` |
| 文本切分器 | `langchain` 内置 | `langchain-text-splitters` |

## RAG 数据流

```text
文件
  ↓
LangChain Community Loader
  ↓
Document + metadata
  ↓
RecursiveCharacterTextSplitter
  ↓
Document Chunk
  ↓
Embedding 模型
  ↓
ChromaDB
  ↓
相似度检索
  ↓
RetrievalQA / ConversationalRetrievalChain
  ↓
LLM 返回答案
```

## 配置和安全

| 项目 | 处理方式 |
| --- | --- |
| API Key | 保存在本地 `.env` |
| Git 跟踪 | `.env` 已加入 `.gitignore` |
| 示例配置 | 使用 `.env.example` |
| 向量库路径 | 使用项目内相对路径，例如 `./vector_db/chroma` |
| Python 环境 | 推荐 Python 3.11 的独立 Conda 环境 |

## 尚未完成的迁移

| 项目 | 当前状态 |
| --- | --- |
| Loader 导入路径 | 已迁移 |
| Chroma 导入路径 | 已迁移 |
| Embedding 导入路径 | 已迁移 |
| 自定义 LLM 基类导入 | 已迁移 |
| Gradio 参数兼容 | 已移除新版不支持的显示参数 |
| Chain 调用方式 | 仍需确认是否从 `chain({...})` 改为 `chain.invoke({...})` |
| 旧版 Chain 行为 | 依赖 `langchain-classic`，后续可改为新版 Runnable 架构 |
