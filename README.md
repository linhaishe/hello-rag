# 个人知识库助手项目

![img](https://picgocloud.com/m/0c09a55f-684e-4530-8b8b-e6f1daa403f8.png)

## 项目背景介绍

是根据datawhale的案例进行的学习记录开发，由于langchain发展过快，这个项目并不会使用原教程里的技术栈版本，会使用近期新的`langchain 1.4.0`版本，并接入了`gemini api` , `python==3.11.15`, 本地 `ChromaDB`

```python
# 创建 Conda 环境
conda create -n llm-universe python==3.11.15
# 激活 Conda 环境
conda activate llm-universe
# 安装依赖项
pip install -r requirements-new.txt
```

```py
# 启动服务为本地 API(Fast API)
uvicorn serve.api:app --reload

# 运行项目
python serve/run_gradio.py
```
```python
python==3.11.15
langchain==1.4.0
langchain-community>=0.3
langsmith>=0.3.45,<1
```

`python -m pip freeze > requirements-new.txt`

| 术语 | 含义 |
|---|---|
| Embedding | 把文本转换为向量的模型或过程 |
| 向量检索 | 根据向量相似度查找文档 |
| Embedding 检索 | 强调使用 Embedding 生成向量后进行的检索 |
| Chroma | 保存向量并执行相似度搜索的向量数据库 |

"../" 是相对于你运行命令时的当前工作目录，而不是相对于当前 .py 文件。

当前 RAG 策略可以描述为：基于 Embedding 和 Chroma 的单路向量相似度检索。

流程：

```
用户问题
→ 问题向量化
→ Chroma 相似度搜索
→ 返回 Top-K 文档块
→ 拼接 context
→ Prompt
→ Gemini / 其他 LLM
→ 最终答案
```

## RAG Process

这个项目的 RAG 分为两条链路：先把文件构建为本地向量库，再用用户问题检索相关内容并交给 LLM 生成答案。

### 1. 文件到向量库：知识库构建链路

```text
Gradio 上传文件或指定 knowledge_db 目录 init_db = gr.Button("知识库文件向量化")
  ↓
create_db_info()
  ↓
create_db()
  ↓
file_loader() 按扩展名选择 Loader
  ↓
loader.load() 读取文件，得到 Document
  ↓
RecursiveCharacterTextSplitter 切分 Document
  ↓
get_embedding() 创建 Embedding 模型
  ↓
Chroma.from_documents() 为每个 Chunk 生成向量
  ↓
保存到 ./vector_db/chroma
```

具体处理方式：

| 文件类型 | Loader | 作用 |
| --- | --- | --- |
| `.pdf` | `PyMuPDFLoader` | 读取 PDF 页面文本和页面 metadata |
| `.md` | `UnstructuredMarkdownLoader` | 使用 Unstructured 解析 Markdown |
| `.txt` | `UnstructuredFileLoader` | 读取普通文本 |

`loader.load()` 读取后得到的是 `Document` 列表，每个对象主要包含：

```python
Document(
    page_content="文件中的文本",
    metadata={"source": "文件路径", "page": 1},
)
```

接着使用：

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
)
```

把较长的 `Document` 切分成多个较小的 Chunk。`chunk_overlap=150` 会让相邻 Chunk 保留部分重复文本，降低关键信息刚好被切断的影响。切分后的 Chunk 仍然保留原文的 metadata。

### 2. 向量化和本地存储

`embedding/call_embedding.py` 中的 `get_embedding()` 负责返回 Embedding 模型对象：

```python
embeddings = get_embedding("m3e")
# HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
```

它本身还没有处理文档。真正的向量化发生在：

```python
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./vector_db/chroma",
)
```

Chroma 会对每个 Chunk 调用 Embedding 模型的 `embed_documents()`，并保存：

```text
Chunk 原文 + metadata + 对应向量
```

当前项目使用本地持久化 ChromaDB。新版 Chroma 在指定 `persist_directory` 后会自动保存，不需要再调用 `vectordb.persist()`。

### 3. 用户问题到最终答案：问答链路

```text
用户输入问题
  ↓
QA_chain_self.answer() 或 Chat_QA_chain_self.answer()
  ↓
retriever 查询 ChromaDB
  ↓
Embedding 模型将问题转换为 Query Vector
  ↓
Chroma 按相似度返回 Top-K 个 Chunk
  ↓
format_docs() 拼接检索到的文本
  ↓
PromptTemplate 组合 context、question 和 chat_history
  ↓
LLM 生成答案
  ↓
StrOutputParser 转换为字符串
  ↓
Gradio 页面或 API 返回答案
```

新版 Runnable 链的结构是：

```python
qa = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": format_history,
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = qa.invoke(question)
```

```
# 这一步内部完成
用户问题 → 向量检索 → 得到相关文档 → 拼接 Prompt → LLM 生成最终答案 → answer
```

这里的 `|` 表示把多个处理步骤连接成一条流水线：

```text
输入问题
→ 检索器
→ Prompt
→ LLM
→ 输出解析器
→ 最终答案
```

### 4. 单轮和多轮问答的区别

| 模式 | 是否检索向量库 | 是否携带历史对话 | 主要入口 |
| --- | --- | --- | --- |
| `Chat with llm` | 否 | 是，使用界面历史拼接 Prompt | `respond()` |
| `Chat db without history` | 是 | 否 | `QA_chain_self` |
| `Chat db with history` | 是 | 是 | `Chat_QA_chain_self` |

无论哪种模式，最终生成自然语言答案的都是 LLM。ChromaDB 只负责保存向量和返回相关文档片段，不负责生成答案。

### Loader 读取

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

### 数据清洗

这个项目暂时还有使用到数据清洗这一步

https://github.com/datawhalechina/llm-universe/blob/main/docs/C3/C3.md#333-%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97

我们期望知识库的数据尽量是有序的、优质的、精简的，因此我们要删除低质量的、甚至影响理解的文本数据。
可以看到上文中读取的pdf文件不仅将一句话按照原文的分行添加了换行符`\n`，也在原本两个符号中间插入了`\n`，我们可以使用正则表达式匹配并删除掉`\n`。

### TextSplitter 切分

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

### Embedding 向量化

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

### ChromaDB 存储

向量存储逻辑在：`database/create_db.py`

`Chroma.from_documents` 会调用传入的`embedding`模型进行`embedding`

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

### 用户提问/相似度检索

`Chat with llm`：问题 → LLM

`Chat db without history`：问题 → 向量检索 → LLM

`Chat db with history`：问题 + 历史 → 向量检索 → LLM

| 按钮                    | 是否查向量库 | 用途                 |
| ----------------------- | ------------ | -------------------- |
| Chat with llm           | 否           | 普通知识问答         |
| Chat db without history | 是           | 基于知识库的单轮问答 |
| Chat db with history    | 是           | 基于知识库的多轮问答 |

| 选项      | 类型                 | 运行位置 | 是否需要 API Key | 特点                                             |
| --------- | -------------------- | -------- | ---------------- | ------------------------------------------------ |
| `openai`  | OpenAI Embedding API | 云端     | 需要             | 效果稳定，接入简单，但按调用量收费               |
| `m3e`     | M3E 本地向量模型     | 本地     | 不需要           | 适合中文和离线场景，但需要下载模型并占用本地资源 |
| `zhipuai` | 智谱 Embedding API   | 云端     | 需要             | 中文支持较好，依赖智谱 API 和网络                |

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

### LLM 生成答案

```
elif provider == "gemini":
    if api_key is None:
        api_key = parse_llm_api_key("gemini")
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
```

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

answer = qa.invoke(question)
answer = re.sub(r"\\n", '<br/>', answer)
self.chat_history.append((question, answer))  # 更新历史记录
```



# QA

## tempfile

程序运行时，系统会创建一个临时文件

```python
import tempfile

with tempfile.NamedTemporaryFile() as file:
    file.write(b"hello")
    print(file.name)
```

## create_db()

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

## extend 和 append

```py
items = [1, 2]
items.append([3, 4])

print(items) # [1, 2, [3, 4]]

items = [1, 2]
items.extend([3, 4])

print(items) # [1, 2, 3, 4]
```

## Runnable

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

因此第一段并行执行后的结果大致是：

```
{
    "context": "检索到的文档内容……",
    "question": "RAG 是什么？",
    "chat_history": "之前的对话……",
}
```

然后继续执行：

```
| prompt
```

把字典填入 Prompt：

```
context       → {context}
question      → {question}
chat_history  → {chat_history}
```

再继续：

```
| llm
```

把完整 Prompt 交给大模型。

最后：

```
| StrOutputParser()
```

把模型返回的消息对象转换成普通字符串。

整体流程：

```
qa.invoke(question)
        ↓
并行准备 context、question、chat_history
        ↓
填充 prompt
        ↓
调用 llm
        ↓
转换成字符串
        ↓
返回答案
```

所以第一段不是“第一个链单独调用”，而是 `qa.invoke(question)` 时，字典中的三个分支会一起执行。

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

## Gradio

```python
llm_btn.click(
   respond,
   inputs=[msg, chatbot, llm, history_len, temperature],
   outputs=[msg, chatbot],
   show_progress="minimal",
)
```

`outputs` 表示：函数执行完成后，把返回值显示或写回哪些 Gradio 组件。

`inputs` 不是自定义参数，而是 Gradio 的固定配置项。

它的作用是：告诉 Gradio，点击按钮时，要把哪些界面组件的值传给函数。

## class Model_center

```python
class Model_center:
    """
    存储问答 Chain 的对象

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """

    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(
        self,
        question: str,
        chat_history: list = [],
        model: str = "openai",
        embedding: str = "openai",
        temperature: float = 0.0,
        top_k: int = 4,
        history_len: int = 3,
        file_path: str = DEFAULT_DB_PATH,
        persist_path: str = DEFAULT_PERSIST_PATH,
    ):
        """
        使用带历史记录的 RAG 问答链回答用户问题。

        Args:
            question: 用户当前输入的问题。
            chat_history: 当前对话历史记录。
            model: 使用的 LLM 模型名称，例如 "openai" 或 Gemini 模型。
            embedding: 使用的 Embedding 模型名称，例如 "openai" 或 "m3e"。
            temperature: 控制模型回答的随机性，数值越高，回答越随机。
            top_k: 从向量数据库中检索的相关文档数量。
            history_len: 参与本次问答的历史对话轮数。
            file_path: 知识库文件或目录的路径。
            persist_path: 持久化向量数据库的保存路径。

        Returns:
            tuple:
                第一个元素：空字符串，用于清空问题输入框；
                第二个元素：更新后的聊天记录，用于刷新聊天窗口。
        """
        chat_history = messages_to_tuples(chat_history)
        if question == None or len(question) < 1:
            return "", tuples_to_messages(chat_history)
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    chat_history=chat_history,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=embedding,
                )
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", tuples_to_messages(
                chain.answer(question=question, temperature=temperature, top_k=top_k)
            )
        except Exception as e:
            return e, tuples_to_messages(chat_history)

    def qa_chain_self_answer(
        self,
        question: str,
        chat_history: list = [],
        model: str = "openai",
        embedding="openai",
        temperature: float = 0.0,
        top_k: int = 4,
        file_path: str = DEFAULT_DB_PATH,
        persist_path: str = DEFAULT_PERSIST_PATH,
    ):
        """
        调用不带历史记录的问答链进行回答
        """
        chat_history = messages_to_tuples(chat_history)
        if question == None or len(question) < 1:
            return "", tuples_to_messages(chat_history)
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=embedding,
                )
            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append((question, chain.answer(question, temperature, top_k)))
            return "", tuples_to_messages(chat_history)
        except Exception as e:
            return e, tuples_to_messages(chat_history)

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()
```

### self.chat_qa_chain_self  的判断逻辑

它在 `Model_center.__init__()` 中第一次被赋值为空字典：

```
def __init__(self):
    self.chat_qa_chain_self = {}
```

之后，在判断成立时再放入问答链：

```
key = (model, embedding)

if key not in self.chat_qa_chain_self:
    self.chat_qa_chain_self[key] = Chat_QA_chain_self(
        model=model,
        embedding=embedding,
        ...
    )
```

完整过程：

第一次调用：

```
self.chat_qa_chain_self = {}
key = ("gemini-3.1-flash-lite", "m3e")
```

判断：

```
key not in {}
```

结果为 `True`，于是创建并保存：

```
self.chat_qa_chain_self[
    ("gemini-3.1-flash-lite", "m3e")
] = Chat_QA_chain_self(...)
```

此时字典大致是：

```
{
    ("gemini-3.1-flash-lite", "m3e"): Chat_QA_chain_self对象
}
```

第二次使用同样的模型组合：

```
key not in self.chat_qa_chain_self
```

结果为 `False`，因为这个 key 已经存在，于是直接复用：

```
chain = self.chat_qa_chain_self[key]
```

所以：

```
__init__：创建空字典
第一次调用：添加模型组合和问答链
后续调用：发现 key 已存在，直接复用
```

如果更换模型组合，例如：

```
("openai", "m3e")
```

这个新 key 不在字典中，就会再创建一条新的问答链。

### 这个为什么用class 而不直接使用def

因为这里需要保存多个问答链对象，并在多次调用之间复用它们。`class` 可以把这些状态保存到 `self` 中。

```
class Model_center:
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}
```

程序启动时创建一个对象：

```
model_center = Model_center()
```

之后每次调用：

```
model_center.chat_qa_chain_self_answer(...)
```

都可以访问之前保存的：

```
self.chat_qa_chain_self
```

第一次使用某个模型组合时创建并保存：

```
self.chat_qa_chain_self[(model, embedding)] = chain
```

下一次使用相同组合时直接复用。

如果直接使用普通 `def`，也可以实现，但需要把缓存字典作为全局变量或参数传来传去：

```
chat_qa_chain_self = {}

def chat_qa_chain_self_answer(...):
    if key not in chat_qa_chain_self:
        chat_qa_chain_self[key] = create_chain(...)
```

现在的 `class` 主要解决两件事：

```
保存状态：self.chat_qa_chain_self
组织相关功能：问答、清空历史等方法放在一起
```

因此这里使用 `class` 是为了保存和管理问答链缓存，不是因为调用函数本身必须用类。

## def tuples_to_messages(history)

页面上看起来像“追加一条消息”，实际上函数每次都会返回完整的聊天记录。

```
return "", tuples_to_messages(
    chain.answer(...)
)
```

例如原来是：

```
[
    {"role": "user", "content": "问题1"},
    {"role": "assistant", "content": "答案1"},
]
```

新增问题后，返回的是完整列表：

```
[
    {"role": "user", "content": "问题1"},
    {"role": "assistant", "content": "答案1"},
    {"role": "user", "content": "问题2"},
    {"role": "assistant", "content": "答案2"},
]
```

Gradio 用这个完整列表重新刷新 `chatbot`，所以视觉上像是只增加了新消息。

不过“缓存”主要影响的是问答链对象：

```
self.chat_qa_chain_self[(model, embedding)]
```

同一个模型组合只创建一次 `Chat_QA_chain_self`。之后新的问题会追加到这个对象内部的：

```
chain.chat_history
```

因此当前流程是：

```
页面发送完整聊天记录
→ 转成 tuples
→ 问答链检索并生成答案
→ 追加到 chain.chat_history
→ 返回完整聊天记录
→ Gradio 刷新整个聊天窗口
```

补充一点：当前代码中，`history_len` 并没有真正限制页面展示数量；它主要应该用于限制传给 Prompt 的历史记录。当前实现实际可能会把缓存对象中的全部历史都用于回答。

Demo 这样处理很常见，但生产系统通常不会只依赖这种内存缓存。

常见区别：

| Demo 做法                       | 实际业务做法                   |
| ------------------------------- | ------------------------------ |
| 用 `self.chat_history` 保存历史 | 按用户、会话 ID 保存历史       |
| 直接返回完整聊天记录            | 前端可以增量追加，或分页加载   |
| 字典缓存问答链对象              | 使用有过期时间和容量限制的缓存 |
| 历史记录保存在进程内存          | 保存到 Redis、数据库或消息存储 |
| 所有历史都传给 Prompt           | 只传最近几轮，或先总结历史     |
| 进程重启后历史丢失              | 历史可以恢复                   |
| 多用户容易混在一起              | 每个用户独立隔离               |

生产环境一般会区分三种数据：

```
前端展示历史
    ↓
会话历史存储
    ↓
提供给 Prompt 的上下文
```

例如：

```
def answer(session_id: str, question: str):
    history = history_store.get(session_id)

    recent_history = history[-5:]

    answer = qa_chain.invoke({
        "question": question,
        "chat_history": recent_history,
    })

    history_store.append(session_id, question, answer)

    return answer
```

当前 Demo 的主要问题是：

```
self.chat_qa_chain_self[(model, embedding)]
```

只按模型和 Embedding 区分，没有按用户或会话区分。多个用户使用相同模型时，可能共用同一个 `chat_history`，导致对话串线。

所以实际工程通常至少要改成：

```
self.chat_qa_chain_self[(session_id, model, embedding)]
```

更进一步则是：

- 聊天历史放 Redis 或数据库；
- 只取最近几轮历史；
- 使用摘要压缩更早的对话；
- 对缓存设置过期时间；
- 前端按消息 ID 增量更新。

这个 Demo 的思路适合学习 Gradio 和 RAG 流程，但不适合直接作为多用户生产架构。

## rich

你想用的应该是 `rich`，它可以格式化终端输出。

项目里已经有 `rich` 依赖，可以直接使用：

```
from rich import print

print("[green]模型加载成功[/green]")
print(f"[cyan]当前模型：{model}[/cyan]")
```

如果想打印字典、列表等结构化数据：

```
from rich import print

print(self.chat_qa_chain_self)
```

更适合调试复杂对象的是 `rich.pretty`：

```
from rich.pretty import pprint

pprint(self.chat_qa_chain_self)
```

如果想要表格：

```
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="已加载的问答链")

table.add_column("模型")
table.add_column("Embedding")

for model, embedding in self.chat_qa_chain_self:
    table.add_row(model, embedding)

console.print(table)
```

简单打印用：

```
from rich import print
```

```python
pip install rich
```

### 带标识

`pprint()` 通常只接收一个要打印的对象：

```
from rich.pretty import pprint

pprint(("chain", chain))
```

如果只是想加提示，推荐使用 `rich.print`：

```
from rich import print

print("[bold cyan]chain:[/bold cyan]", chain)
```

或者普通 Python：

```
print("chain:", chain)
```

调试时最简单的是：

```
from rich.pretty import pprint

pprint({"chain": chain})
```

这样输出会带有明确的 `chain` 标识。

## docker 查看服务日志

`docker compose logs -f gradio`

## 容器内跑了向量存储，本地也有新的数据库产生了
这是因为 Compose 配置了目录挂载：

```yaml
volumes:
  - ./vector_db:/app/vector_db
```

它的含义是：

```text
本地 ./vector_db
↔ 容器内 /app/vector_db
```

所以程序虽然在容器里写入：

```text
/app/vector_db
```

文件实际上会同步保存到本地：

```text
./vector_db
```

这是为了防止容器删除后向量数据库丢失，属于正常现象。

如果你希望数据只存在容器中，可以删除这段：

```yaml
volumes:
  - ./vector_db:/app/vector_db
```

但这样容器删除或重建后，向量数据库也会丢失。

更推荐保留当前配置：

```text
容器：运行程序
本地：持久化向量数据库
```

如果不想直接看到项目目录中的数据库，也可以改用 Docker 命名卷：

```yaml
volumes:
  chroma_data:

services:
  api:
    volumes:
      - chroma_data:/app/vector_db

  gradio:
    volumes:
      - chroma_data:/app/vector_db
```

这样数据仍然在宿主机上，但由 Docker 管理，不会出现在项目的 `vector_db` 目录中。

## Container hello-rag-gradio-1 这个后面自动-1
`hello-rag-gradio-1` 是 Docker Compose 自动生成的容器名称，通常格式是：

```text
项目名-服务名-序号
```

当前对应关系：

```text
hello-rag  → Compose 项目名
gradio     → docker-compose.yml 中的服务名
1          → 第一个实例
```

所以：

```text
hello-rag-gradio-1
```

表示 `hello-rag` 项目中的第一个 `gradio` 容器。

如果扩展多个实例，可能会出现：

```text
hello-rag-gradio-2
hello-rag-gradio-3
```

这个 `-1` 是正常的，不是报错。

如果想自定义固定名称，可以写：

```yaml
services:
  gradio:
    container_name: hello-rag-gradio
```

不过一般不建议设置 `container_name`，因为会限制 Compose 扩展多个实例。
它并没有自动扩展。

`-1` 只是 Docker Compose 的实例编号，即使只有一个容器也会加上：

```text
hello-rag-gradio-1
```

如果你执行：

```bash
docker compose up --scale gradio=2
```

才会真正启动多个实例：

```text
hello-rag-gradio-1
hello-rag-gradio-2
```

所以当前的 `-1` 只表示：

```text
第 1 个 gradio 容器
```

不是自动扩容，也不会额外消耗资源。

`docker compose up -d --build `

## 本地的回复质量和docker内有差别

差异主要不是 Docker 本身造成的，而是 Docker 当前加载到的向量库内容和本地不一致。

从你的回答看，Docker 检索到的是：

```text
Joe 写落地灯规格说明书的示例
```

而且还是重复片段。这说明当前 `vector_db/chroma` 中存在重复或旧的向量数据，`top_k=3` 可能全部命中了同一个示例，因此模型没有拿到整篇文章的上下文。

项目当前的问题是：

```python
Chroma.from_documents(...)
```

每次点击“知识库文件向量化”都会继续往原来的 Chroma 中追加数据，不会自动清空旧数据。

建议重新建立一次干净的向量库：

```bash
docker compose down
```

删除本地旧向量库中的内容：

```bash
rm -rf ./vector_db/chroma
```

然后重新启动：

```bash
docker compose up -d --build
```

再在 Gradio 中：

1. 选择 `m3e`；
2. 上传或选择完整知识库；
3. 点击“知识库文件向量化”一次；
4. 将 `top_k` 调到 `8` 或 `10`；
5. 使用 `Chat db without history` 查询。

另外，确认本地和 Docker 使用的是同一份数据库：

```bash
ls -lah ./vector_db/chroma
# 检查容器内向量库：
`docker compose exec gradio ls -lah /app/vector_db/chroma`

# 检查本地向量库：
`ls -lah ./vector_db/chroma`

# 两边应该看到同一份 `chroma.sqlite3` 和索引文件。
```

如果希望进一步确认检索结果，可以在 `qa` 链中临时打印：

```python
docs = retriever.invoke(question)
pprint([
    {
        "source": doc.metadata.get("source"),
        "content": doc.page_content[:200],
    }
    for doc in docs
])
```






