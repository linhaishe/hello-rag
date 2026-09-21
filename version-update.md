## 本次依赖升级记录

### 升级背景

原项目使用较旧的 LangChain 导入路径和 Chroma 依赖。旧版 `chromadb==0.3.29` 要求 `pydantic<2`，而新版 LangChain 需要 Pydantic 2，因此会产生依赖冲突。

### 主要变化

| 功能 | 原写法 | 新写法 |
| --- | --- | --- |
| Prompt | `langchain.prompts` | `langchain_core.prompts` |
| Chain | `langchain.chains` | `langchain_classic.chains` |
| Memory | `langchain.memory` | `langchain_classic.memory` |
| Chroma | `langchain.vectorstores` | `langchain_chroma` |
| OpenAI Chat 模型 | `langchain.chat_models` | `langchain_openai` |

### 代码修改

涉及文件：

```text
database/create_db.py
qa_chain/QA_chain_self.py
qa_chain/Chat_QA_chain_self.py
qa_chain/model_to_llm.py
```

新版导入示例：

```python
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
```

### 依赖变化

新版环境需要这些 LangChain 集成包：

```bash
python -m pip install \
  langchain langchain-core langchain-community \
  langchain-classic langchain-chroma langchain-openai \
  langchain-huggingface chromadb sentence-transformers
```

项目还需要：

```bash
python -m pip install \
  fastapi uvicorn gradio python-dotenv \
  unstructured pymupdf requests openai zhipuai websocket-client
```

### 安装建议

建议在 Python 3.11 的干净 Conda 环境中安装：

```bash
conda create -n hello-rag-new python=3.11
conda activate hello-rag-new
python -m pip install -U pip setuptools wheel
python -m pip check
```

旧版依赖清单已保留为 `requirements-old.txt`，不要再使用其中的旧版 Chroma 约束安装新版 LangChain。

### 配置文件安全

`.env` 用于保存本地 API Key，已加入 `.gitignore`，不会继续被 Git 跟踪。可以参考 `.env.example` 创建本地配置。

### 尚未完成的升级

部分文件仍使用旧版 Loader 和 Embedding 导入路径，后续可能需要迁移到：

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
```

旧版 Chain 调用方式也可能需要从：

```python
chain({...})
```

迁移为：

```python
chain.invoke({...})
```
