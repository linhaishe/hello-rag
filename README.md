# 个人知识库助手项目

# 一、引言

## 1、项目背景介绍

是根据datawhale的案例进行的学习记录开发，由于langchain发展过快，这个项目并不会使用原教程里的技术栈版本，会使用近期新的langchain版本。
这个项目使用的是本地 ChromaDB

```python
# 创建 Conda 环境
conda create -n llm-universe python==3.11.15
# 激活 Conda 环境
conda activate llm-universe
# 安装依赖项
pip install -r requirements.txt
```

```py
# 启动服务为本地 API(Fast API)
uvicorn serve.api:app --reload

# 运行项目
python serve/run_gradio.py -model_name='chatglm_std' -embedding_model='m3e' -db_path='./data_base/knowledge_db' -persist_path='./data_base/vector_db'
```
```python
python==3.11.15
langchain==1.4.0
langchain-community>=0.3
langsmith>=0.3.45,<1
```

`python -m pip freeze > requirements-new.txt`

Mac 是 Apple 芯片（osx-arm64），而 defaults 源里没有适用于 Apple 芯片的 Python 3.9.0，所以创建失败。
直接执行：

```bash
conda create -n llm-universe -c conda-forge python=3.9
conda activate llm-universe
python --version
```

如果显示类似：`Python 3.9.x` 就成功了。如果项目只要求 `Python 3.9`，不要求必须是 `3.9.0`，这就是最简单的解决办法。

```python
database/create_db.py
qa_chain/get_vectordb.py
qa_chain/QA_chain_self.py
embedding/call_embedding.py
serve/run_gradio.py

→ Loader 读取
→ TextSplitter 切分
→ Embedding 向量化
→ ChromaDB 存储
→ 相似度检索
→ LLM 生成答案
```

"../" 是相对于你运行命令时的当前工作目录，而不是相对于当前 .py 文件。

# RAG Process

## 流程

```python
文档 / loader
  ↓
Docling / MinerU / Unstructured (三选一)
  ↓
提取标题、段落、表格、公式
  ↓
LlamaIndex NodeParser
  ↓
切分成 Node / Chunk，并保留 metadata
  ↓
Sentence Transformers
  ↓
为每个 Chunk 生成 Embedding
  ↓
ChromaDB / FAISS
  ↓
保存：向量 + 原文 + metadata
```

```py
用户问题
  ↓
Sentence Transformers
  ↓
把问题转换成 Query Embedding
  ↓
ChromaDB / FAISS
  ↓
检索相似 Chunk
  ↓
可选：Reranker 精排
  ↓
拼接 Prompt
  ↓
LLM 回答
```

## Loader 读取

基本可以理解为“读取文件”，但不只是拿到文件路径。通常包括：

1. 根据文件路径找到文件
2. 打开文件
3. 读取文件内容
4. 转换成程序能处理的文档对象
5. 附带一些 metadata，例如文件名、来源、页码

```py
# database/create_db.py

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
# UnstructuredMarkdownLoader 底层就是调用 unstructured 包来解析 Markdown 文件。

# 获取文件路径
def get_files(dir_path):
    file_list = [] # [database/sub/b.pdf]
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
    return file_list


def file_loader(file, loaders: list):
    """
    根据传入的文件或目录，找到支持的文件，并创建对应的 Loader 放入 loaders 列表。
      传入文件/目录
      ↓
      如果是临时文件，获取真实路径
      ↓
      如果是目录，递归遍历里面的内容
      ↓
      如果是文件，判断扩展名
      ↓
      创建对应 Loader
      ↓
      加入 loaders 列表
    
    参数:
    file: 存放文件的目录或文件。
    loaders: 用于存放Loader的list
    """
    # 判断 file 是否为临时文件对象(程序运行期间临时创建的文件，用完后通常会自动删除),项目中的 Gradio 上传文件可能会先保存成临时文件，因此代码需要取出真实路径,转换后，file 就从“临时文件对象”变成了普通文件路径字符串
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    
    # 判断 file 是否是一个真实存在的文件，获取临时文件的真实路径
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
        return
    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        #  loaders.append 先把“文件加载器对象”放进列表
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        pattern = r"不存在|风控" # 检查 Markdown 文件路径中是否包含“不存在”或“风控”
        match = re.search(pattern, file)
        if not match:
            loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    return
```

## 数据清洗

## TextSplitter 切分

文档切分器 Text Splitters

Refs: https://github.com/linhaishe/FEnotes/blob/main/ai-coding/LangChain/10-RAG.md#234-%E5%85%B7%E4%BD%93%E5%AE%9E%E7%8E%B0

① CharacterTextSplitter：Split by character

② RecursiveCharacterTextSplitter：最常用

③ TokenTextSplitter/CharacterTextSplitter：Split by tokens

④ SemanticChunker：语义分块

⑤ HTMLHeaderTextSplitter(了解)

⑥ CodeTextSplitter(了解)

⑦ MarkdownTextSplitter(了解)

- RecursiveCharacterTextSplitter(): 按字符串分割文本，递归地尝试按不同的分隔符进行分割文本。
- CharacterTextSplitter(): 按字符来分割文本。
- MarkdownHeaderTextSplitter(): 基于指定的标题来分割markdown 文件。
- TokenTextSplitter(): 按token来分割文本。
- SentenceTransformersTokenTextSplitter(): 按token来分割文本
- Language(): 用于 CPP、Python、Ruby、Markdown 等。
- NLTKTextSplitter(): 使用 NLTK（自然语言工具包）按句子分割文本。
- SpacyTextSplitter(): 使用 Spacy按句子的切割文本。

```py
# database/create_db.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

loaders = []
[file_loader(file, loaders) for file in files]
docs = []
for loader in loaders:
    if loader is not None:
        docs.extend(loader.load())
# 创建“切分规则”
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
# 切分文档
split_docs = text_splitter.split_documents(docs)
```

切分文本、得到切分后的文档、准备 Embedding 模型

```py
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
```

`text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)`

创建文本切分器：

- `chunk_size` 每个文本块最多约 `500` 个字符
- `chunk_overlap` 相邻文本块重叠 `150` 个字符
- 重叠部分可以避免上下文在切分处丢失。

`split_docs = text_splitter.split_documents(docs)`

把原始文档列表 `docs` 切分成更小的文档块。`split_docs` 仍然是 `Document` 列表，每个对象通常包含：

```py
Document(
    page_content="切分后的文本",
    metadata={
        "source": "文件路径",
        "page": 1
    }
)
```

## Embedding 向量化

常用嵌入模型：

| 模型                     | 机构                   | 描述                                       |
| ------------------------ | ---------------------- | ------------------------------------------ |
| `bge-large-zh`           | 北京智源研究院（BAAI） | 开源，向量维度 1024，序列长度 512          |
| `bge-base-zh`            | BAAI                   | 开源，向量维度 768，序列长度 512           |
| `bge-small-zh`           | BAAI                   | 开源，向量维度 512，序列长度 512           |
| `bge-m3`                 | BAAI                   | 开源，多语言，向量维度 1024，序列长度 8192 |
| `text-embedding-3-small` | OpenAI                 | 多语言，向量维度 1536，序列长度 8192       |
| `text-embedding-3-large` | OpenAI                 | 多语言，向量维度 3072，序列长度 8192       |

LangChain中针对向量化模型的封装提供了两种接口，一种针对句子的向量化embed_query ，一种针对文档的向量化(embed_documents) 。

**Embedding（嵌入）**：把文本、图片等对象转换成数字表示的方法或过程。

**Vector（向量）**：转换后得到的一串数字。

例如：

```
“今天天气很好”
→ [0.12, -0.38, 0.77, ...]
```

这里：

- 转换过程叫 **Embedding**
- 得到的数字数组叫 **Vector**

`get_embedding()` in `embedding/call_embedding.py` 是用来创建 Embedding 模型对象的。

```python
def get_embedding(embedding: str, ...):
    if embedding == "m3e":
        return HuggingFaceEmbeddings(
            model_name="moka-ai/m3e-base"
        )
```

它本身不直接把文档转换成向量，而是返回一个“Embedding 模型”。

真正转换发生在：

```python
# database/create_db.py
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings
)
```

流程是：

```
get_embedding("m3e")
→ 创建 m3e Embedding 模型
→ Chroma.from_documents(...)
→ 对每个 Document Chunk 调用 Embedding
→ 生成向量并保存到 ChromaDB
```

也可以用于查询文本：

```
embeddings.embed_query("你好")
```

- `get_embedding()`：获取 Embedding 模型
- `embed_documents()`：将多个文档转成向量
- `embed_query()`：将问题转成向量
- `Chroma`：保存和检索这些向量

基本理解正确，但更准确地说：

```
Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings
)
```

`Chroma` 本身负责：

- 接收文档
- 调用你传入的 `embedding` 模型
- 保存向量、原文和 metadata
- 创建向量数据库

真正负责生成向量的是：

```
embedding=embeddings
```

例如：

```
embeddings = HuggingFaceEmbeddings(
    model_name="moka-ai/m3e-base"
)
```

流程是：

```
split_docs
↓
Chroma.from_documents(...)
↓
调用 embeddings.embed_documents(...)
↓
得到向量
↓
保存到 ChromaDB
```

所以不是 Chroma 内置固定的 Embedding 模型，而是：

> Chroma 提供存储流程，并使用你传入的 Embedding 模型生成向量。

也可以传入 OpenAI Embedding：

```
embeddings = OpenAIEmbeddings()
```

或者其他兼容 LangChain Embeddings 接口的模型。

## ChromaDB 存储

向量存储逻辑在：`database/create_db.py`

```python
# 核心代码：
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./vector_db/chroma"
)
```

```
split_docs
→ Chroma.from_documents()
→ Embedding 向量化
→ 保存到 ./vector_db/chroma
```

随后：

```
vectordb.persist()
```

将数据持久化到磁盘。

重新加载已有数据库的位置：

```python
def load_knowledge_db(path, embeddings):
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb
```

注意：当前 `create_db()` 里把路径写死为：

```
"./vector_db/chroma"
```

所以传入的 `persist_directory` 参数实际上没有被使用。 short fix 可以改成：

```python
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
```

## 用户提问/相似度检索

`Chat with llm`：问题 → LLM

`Chat db without history`：问题 → 向量检索 → LLM

`Chat db with history`：问题 + 历史 → 向量检索 → LLM

| 按钮                    | 是否查向量库 | 用途                 |
| ----------------------- | ------------ | -------------------- |
| Chat with llm           | 否           | 普通知识问答         |
| Chat db without history | 是           | 基于知识库的单轮问答 |
| Chat db with history    | 是           | 基于知识库的多轮问答 |

查寻流程

```py
点击 Chat db without history
→ QA_chain_self
→ get_vectordb()
→ Chroma 加载向量库
→ retriever.invoke(question)
→ 找到相关文档
→ Prompt + 上下文
→ LLM
→ 返回答案
```

代码路径大致是：

```
db_wo_his_btn
→ qa_chain_self_answer()
→ QA_chain_self.answer()
→ retriever
→ self.qa_chain.invoke()
→ self.llm
→ 返回答案
```

## LLM 生成答案

# tempfile

程序运行时，系统会创建一个临时文件

```python
import tempfile

with tempfile.NamedTemporaryFile() as file:
    file.write(b"hello")
    print(file.name)
```

# create_db.py

```python
if not os.path.isfile(file):
    # list 会循环调用 file_loader()
    [file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
    return
```

```py
# equals
for f in os.listdir(file):
    file_loader(os.path.join(file, f), loaders)
```
1. `os.listdir(file)` 获取目录中的所有文件名和子目录名，只返回名称，不返回完整路径。 / `['a.txt', 'sub']`
2. `os.path.join(file, f)` 拼接出完整路径 `os.path.join("data", "a.txt")`

```
project/
└── data/
    ├── a.txt
    └── sub/
        └── b.pdf
```

```py
import os

file = "data"

names = os.listdir(file)
print(names)

for f in names:
    full_path = os.path.join(file, f)
    print(full_path)
    
# ['a.txt', 'sub']
# data/a.txt
# data/sub
```

# extend 和 append

```py
items = [1, 2]
items.append([3, 4])

print(items) # [1, 2, [3, 4]]

items = [1, 2]
items.extend([3, 4])

print(items) # [1, 2, 3, 4]
```

# Runnable

Runnable 可以理解成：

> 一个“可以被调用的处理步骤”，并且多个步骤可以通过 `|` 连接成流水线。前一个的输出是后一个的输入/参数

```py
qa = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(format_history),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = qa.invoke(question)
```

最简单的例子：

```py
from langchain_core.runnables import RunnableLambda

double = RunnableLambda(lambda x: x * 2)

double.invoke(3)
# 6
```

### `|` 是什么？

```
chain = step1 | step2 | step3
```

表示：

```
输入
→ step1
→ step2
→ step3
→ 输出
```

调用时只需要：

```
chain.invoke(input)
```

不是每个步骤都手动调用。

### 你项目里的 Runnable

```
qa = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(format_history),
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

调用：

```
answer = qa.invoke(question)
```

内部流程：

```
问题
├─ retriever：查询 ChromaDB
│    ↓
│  format_docs：整理检索结果
│
├─ RunnablePassthrough：原样保留问题
│
└─ format_history：整理聊天历史
        ↓
得到 prompt 所需的三个变量
        ↓
PromptTemplate
        ↓
LLM
        ↓
StrOutputParser
        ↓
字符串答案
```

这个字典：

```
{
    "context": ...,
    "question": ...,
    "chat_history": ...
}
```

不是普通的数据收集，而是 Runnable Mapping。三个分支会根据同一个输入生成 Prompt 的变量。

### 常见 Runnable

```
RunnableLambda(func)
```

把普通 Python 函数包装成 Runnable。

```
RunnablePassthrough()
```

原样返回输入。

```
prompt
```

PromptTemplate 本身也是 Runnable。

```
llm
```

聊天模型本身也是 Runnable。

```
StrOutputParser()
```

把模型输出转换成字符串。

`Runnable` 统一提供：

```
.invoke(input)       # 同步调用
.ainvoke(input)      # 异步调用
.batch(inputs)       # 批量调用
.stream(input)       # 流式输出
```

官方文档中的 RAG 示例也使用了类似结构：`retriever | format_docs`、`RunnablePassthrough()`，再通过 `chain.invoke(...)` 执行整条链。[LangChain Runnable 示例](https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.tavily_search_api.TavilySearchAPIRetriever.html)

推荐阅读顺序：

1. [Runnable 接口与基本调用](https://python.langchain.com/docs/concepts/runnables/)
2. [RunnablePassthrough 和 RunnableLambda](https://python.langchain.com/docs/how_to/passthrough/)
3. [Runnable 与 LCEL 管道表达式](https://python.langchain.com/docs/concepts/lcel/)
4. [LangChain RAG Runnable 示例](https://python.langchain.com/docs/tutorials/rag/)
5. [Runnable 的批量、异步和流式调用](https://python.langchain.com/docs/concepts/runnables/#invoke)

建议你先掌握这条核心模式：

```
chain = input_mapper | prompt | llm | output_parser
result = chain.invoke(input)
```

这基本就是新版 LangChain 的核心使用方式。
