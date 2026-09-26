import re
from qa_chain.get_vectordb import get_vectordb
from qa_chain.model_to_llm import model_to_llm
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Chat_QA_chain_self:
    """ "
    带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        top_k: int = 4,
        chat_history: list = [],
        file_path: str = None,
        persist_path: str = None,
        appid: str = None,
        api_key: str = None,
        Spark_api_secret: str = None,
        Wenxin_secret_key: str = None,
        embedding="openai",
        embedding_key: str = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        # self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key

        self.vectordb = get_vectordb(
            self.file_path, self.persist_path, self.embedding, self.embedding_key
        )

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    def change_history_length(self, history_len: int = 1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n - history_len :]

    def answer(self, question: str = None, temperature=None, top_k=4):
        """
        核心方法，调用问答链
        arguments:
        - question：用户提问
        """

        if not question:
            return "", self.chat_history  # 返回空消息 + 聊天历史

        if temperature == None:
            temperature = self.temperature
        llm = model_to_llm(
            self.model,
            temperature,
            self.appid,
            self.api_key,
            self.Spark_api_secret,
            self.Wenxin_secret_key,
        )

        # self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # “创建检索器”，还没有真正开始检索
        retriever = self.vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )  # 默认similarity，k=4

        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""使用以下上下文和历史对话回答问题。如果你不知道答案，就说你不知道。
上下文:
{context}
历史对话:
{chat_history}
问题: {question}
回答:""",
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def format_history(_):
            return "\n".join(f"用户: {q}\n助手: {a}" for q, a in self.chat_history)
        # 这一步内部完成:用户问题 → 向量检索 → 得到相关文档 → 拼接 Prompt → LLM 生成最终答案 → answer
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
        answer = re.sub(r"\\n", "<br/>", answer)
        self.chat_history.append((question, answer))  # 更新历史记录

        return self.chat_history  
        # 返回本次回答和更新后的历史记录,把问题和最终答案保存到历史记录中,[(用户问题, 最终答案), ...]
