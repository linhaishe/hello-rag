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

```
python==3.11.15
langchain==1.4.0
langchain-community>=0.3
langsmith>=0.3.45,<1
```

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
