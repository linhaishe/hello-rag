# 评估需要完善的地方

针对当前的 [evaluation/ragas_eval.py](/Users/chenruo/Documents/GitHub/hello-rag/evaluation/ragas_eval.py)，它已经完成了最基本的 RAGAs 评估流程，但还不算完善。

## RAG 评估样本的基本要求

每条样本至少应包含：

```python
{
    "user_input": "用户问题",
    "retrieved_contexts": [
        "检索到的文档片段1",
        "检索到的文档片段2",
    ],
    "response": "RAG 系统生成的答案",
}
```

对应关系：

```text
user_input          → 用户问题
retrieved_contexts  → 实际检索到的上下文
response            → 模型最终答案
```

如果还要评估答案是否正确，建议增加：

```python
{
    "reference": "人工确认的标准答案",
}
```

## 当前脚本已经完成的部分

当前脚本已经做了：

- 加载项目中的 Markdown 知识库；
- 使用 M3E 生成 Embedding；
- 使用 Chroma 进行向量检索；
- 使用 Gemini 生成答案；
- 保存 `user_input`；
- 保存 `retrieved_contexts`；
- 保存 `response`；
- 计算 `Faithfulness`；
- 计算 `AnswerRelevancy`。

因此，它可以用于一次基础质量检查。

## 当前脚本不完善的地方

### 1. 评估问题太少

现在只有 3 个问题：

```python
questions = [
    "文本转换主要解决什么问题？",
    "文本转换有哪些常见方法？",
    "请总结文本转换文章的主要观点和示例。",
]
```

3 条样本不足以代表整个 RAG 系统。

建议至少覆盖：

```text
事实查询
概念解释
文章总结
多步骤问题
不存在信息的问题
代码或专有名词问题
```

例如：

```python
questions = [
    "文本转换主要解决什么问题？",
    "文本转换有哪些常见方法？",
    "请总结文本转换文章的主要观点和示例。",
    "文本转换和文本扩展有什么区别？",
    "文章中是否介绍了翻译任务？",
    "知识库中没有提到的内容是什么？",
]
```

### 2. 没有 `reference`

当前主要评估：

```text
Faithfulness
Answer Relevancy
```

这两个指标不强制要求标准答案，但如果想评估“答案是否正确”，应该增加人工标准答案：

```python
{
    "user_input": "文本转换主要解决什么问题？",
    "reference": "文本转换主要用于改变文本的语言、格式、语气或表达方式。",
}
```

否则只能判断：

```text
答案是否基于上下文
答案是否与问题相关
```

不能充分判断：

```text
答案是否符合人工预期
```

### 3. 没有保存评估结果

当前只是：

```python
print(results)
```

建议保存成 CSV：

```python
results.to_pandas().to_csv(
    "evaluation/ragas_results.csv",
    index=False,
)
```

否则每次运行后，历史结果会丢失，无法比较修改前后的效果。

### 4. 没有记录检索参数

评估结果最好记录：

```text
Embedding 模型
LLM 模型
chunk_size
chunk_overlap
top_k
Prompt 版本
评估时间
```

否则后面看到一个分数时，不知道它对应哪套配置。

### 5. 评估模型和回答模型相同

当前都是 Gemini：

```python
llm = ChatGoogleGenerativeAI(...)
```

它既生成最终答案，也负责评估答案。

这样可能导致评估偏向自己生成的答案。更稳妥的方式是：

```text
回答模型：Gemini
评估模型：另一个模型，或固定版本的评估模型
```

Demo 阶段可以共用，生产评估最好分开。

### 6. 每次运行都会重新构建向量库

当前：

```python
vectorstore = Chroma.from_documents(chunks, embedding_model)
```

每次运行都会重新切分、向量化和创建向量库，速度较慢，也可能造成重复数据。

评估脚本更适合：

```python
vectorstore = Chroma(
    persist_directory="./vector_db/chroma",
    embedding_function=embedding_model,
)
```

直接加载已经构建好的项目向量库。

## 一个更完整的评估样本

```python
{
    "user_input": "请总结文本转换文章的主要观点、方法和示例。",
    "retrieved_contexts": [
        "文本转换可以改变文本的语言、格式和语气。",
        "常见任务包括翻译、改写、格式转换和风格调整。",
    ],
    "response": "文本转换主要用于改变文本的表达形式，包括翻译、改写、格式转换和风格调整。",
    "reference": "文章介绍了文本转换的定义、常见任务和实际示例。",
}
```

## 结论

当前脚本属于：

```text
基础版 RAG 评估 Demo
```

已经可以评估：

```text
Faithfulness
Answer Relevancy
```

但还不适合直接作为生产质量评估。主要需要补充：

```text
更多评估问题
+ reference 标准答案
+ 结果保存
+ 参数记录
+ 直接加载已有向量库
```

简单判断：

```text
当前版本：能运行、能得到指标
完善版本：能比较版本、定位问题、支持回归评估
```