# 个人知识库助手项目

# 一、引言

## 1、项目背景介绍

是根据datawhale的案例进行的学习记录开发，由于langchain发展过快，这个项目并不会使用原教程里的技术栈版本，会使用近期新的langchain版本。

```
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
python run_gradio.py -model_name='chatglm_std' -embedding_model='m3e' -db_path='./data_base/knowledge_db' -persist_path='./data_base/vector_db'

```
python==3.11.15
langchain==1.4.0
langchain-community>=0.3
langsmith>=0.3.45,<1
```

`python -m pip freeze > requirements-new.txt`

Mac 是 Apple 芯片（osx-arm64），而 defaults 源里没有适用于 Apple 芯片的 Python 3.9.0，所以创建失败。
直接执行：
```
conda create -n llm-universe -c conda-forge python=3.9
conda activate llm-universe
python --version
```
如果显示类似：
`Python 3.9.x`
就成功了。
如果项目只要求 `Python 3.9`，不要求必须是 `3.9.0`，这就是最简单的解决办法。

```
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

```
"../" 是相对于你运行命令时的当前工作目录，而不是相对于当前 .py 文件。
```

```py
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

### Loader 读取

基本可以理解为“读取文件”，但不只是拿到文件路径。通常包括：

1. 根据文件路径找到文件
2. 打开文件
3. 读取文件内容
4. 转换成程序能处理的文档对象
5. 附带一些 metadata，例如文件名、来源、页码

```py
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

### TextSplitter 切分

```py
loaders = []
[file_loader(file, loaders) for file in files]
docs = []
for loader in loaders:
    if loader is not None:
        docs.extend(loader.load())
# 创建“切分规则”
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
# 切分文档
split_docs = text_splitter.split_documents(docs)
```

### Embedding 向量化

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

### ChromaDB 存储
### 相似度检索
### LLM 生成答案

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

# 切分文档

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
