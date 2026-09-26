"""使用 RAGAs 评估当前项目的生成质量和检索质量。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    embedding_model = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )

    documents = DirectoryLoader(
        str(PROJECT_ROOT / "knowledge_db/prompt_engineering"),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    ).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
    ).split_documents(documents)

    vectorstore = Chroma.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(
        "只根据上下文回答问题。如果上下文没有答案，请明确说不知道。\n"
        "上下文：{context}\n问题：{question}"
    )
    answer_chain = prompt | llm | StrOutputParser()

    evaluation_cases = [
        {
            "question": "大语言模型可以完成哪些文本转换任务？",
            "reference": (
                "大语言模型可以完成多语言翻译、语种识别、语气与写作风格调整、"
                "结构化数据格式转换，以及拼写和语法纠正。"
            ),
        },
        {
            "question": "为什么大语言模型翻译通常比传统统计机器翻译更自然？",
            "reference": (
                "因为大语言模型能学习不同语言在词汇、语法和语义上的对应关系，"
                "结合上下文理解原句意图并动态调整句子结构，而不是只做逐词替换。"
            ),
        },
        {
            "question": "文章中的文件格式转换示例是如何把 JSON 转换成 HTML 的？",
            "reference": (
                "示例把包含餐厅员工姓名和邮箱的 JSON 数据交给大语言模型，"
                "在提示词中要求保留表格标题和列名并输出 HTML 表格。"
            ),
        },
        {
            "question": "文章的综合样例一次完成了哪些处理步骤？",
            "reference": (
                "综合样例先纠正英文评论的拼写和语法，再翻译成中文，"
                "然后改写成优质淘宝评论，分别给出优点、缺点和总结，"
                "最后以 Markdown 格式输出。"
            ),
        },
    ]
    samples = []
    for case in evaluation_cases:
        question = case["question"]
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        answer = answer_chain.invoke(
            {"context": "\n\n".join(contexts), "question": question}
        )
        samples.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": case["reference"],
            }
        )

    evaluator_llm = LangchainLLMWrapper(llm)
    results = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(
                llm=evaluator_llm,
                embeddings=LangchainEmbeddingsWrapper(embedding_model),
            ),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ],
    )
    print(results)
    score_columns = [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]
    print(results.to_pandas()[score_columns].to_string(index=False))


if __name__ == "__main__":
    main()
