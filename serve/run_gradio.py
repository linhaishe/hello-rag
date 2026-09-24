# 导入必要的库

import re
import sys
import os
from rich.pretty import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_chain.QA_chain_self import QA_chain_self
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from database.create_db import create_db_info
from llm.call_llm import get_completion, LLM_MODEL_DICT
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import io  # 用于处理流式数据（例如文件流）
import IPython.display  # 用于在 IPython 环境中显示数据，例如图片

# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "gemini-3.1-flash-lite"
EMBEDDING_MODEL_LIST = ["zhipuai", "openai", "m3e"]
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db/chroma")
AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "./figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "./figures/datawhale_logo.png"


def messages_to_tuples(messages):
    history = []
    question = None
    for message in messages or []:
        if message["role"] == "user":
            question = message["content"]
        elif message["role"] == "assistant" and question is not None:
            history.append((question, message["content"]))
            question = None
    return history


def tuples_to_messages(history: list[tuple[str, str]]) -> list[dict[str, str]]:
    """
    将问答元组列表转换为 Gradio 使用的消息列表。

    Args:
        history: 问答历史列表。
            每条记录是一个 tuple，格式为
            (用户问题, 助手回答)。

    Returns:
        list[dict[str, str]]: 转换后的消息列表。
            用户消息格式为
            {"role": "user", "content": question}；
            助手消息格式为
            {"role": "assistant", "content": answer}。
    """
    return [
        message
        for question, answer in history
        for message in (
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        )
    ]


def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")


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
                # 把“LLM 模型 + Embedding 模型”这个组合当作唯一标识,把聊天记录存起来
                # some_dic[("gemini-3.1-flash-lite", "m3e")]
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


def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt


def respond(
    message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048
):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    chat_history = messages_to_tuples(chat_history)
    if message == None or len(message) < 1:
        return "", tuples_to_messages(chat_history)
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens
        )
        # 将bot_message中\n换为<br/>
        bot_message = re.sub(r"\\n", "<br/>", bot_message)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", tuples_to_messages(chat_history)
    except Exception as e:
        return e, tuples_to_messages(chat_history)


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        gr.Image(
            value=AIGC_LOGO_PATH,
            scale=1,
            min_width=10,
            show_label=False,
            container=False,
        )

        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>动手学大模型应用开发</center></h1>
                <center>LLM-UNIVERSE</center>
                """)
        gr.Image(
            value=DATAWHALE_LOGO_PATH,
            scale=1,
            min_width=10,
            show_label=False,
            container=False,
        )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=400,
                avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH),
            )
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(components=[chatbot], value="Clear console")

        with gr.Column(scale=1):
            file = gr.File(
                label="请选择知识库目录",
                file_count="directory",
                file_types=[".txt", ".md", ".docx", ".pdf"],
            )
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(
                    0,
                    1,
                    value=0.01,
                    step=0.01,
                    label="llm temperature",
                    interactive=True,
                )

                top_k = gr.Slider(
                    1,
                    10,
                    value=3,
                    step=1,
                    label="vector db search top k",
                    interactive=True,
                )

                history_len = gr.Slider(
                    0, 5, value=3, step=1, label="history length", interactive=True
                )

            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True,
                )

                embeddings = gr.Dropdown(
                    EMBEDDING_MODEL_LIST,
                    label="Embedding model",
                    value=INIT_EMBEDDING_MODEL,
                )

        # 设置初始化向量数据库按钮的点击事件。当点击时，调用 create_db_info 函数，并传入用户的文件和希望使用的 Embedding 模型。
        init_db.click(create_db_info, inputs=[file, embeddings], outputs=[msg])

        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_with_his_btn.click(
            model_center.chat_qa_chain_self_answer,
            inputs=[msg, chatbot, llm, embeddings, temperature, top_k, history_len],
            outputs=[msg, chatbot],
        )
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(
            model_center.qa_chain_self_answer,
            inputs=[msg, chatbot, llm, embeddings, temperature, top_k],
            outputs=[msg, chatbot],
        )
        # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        llm_btn.click(
            respond,
            inputs=[msg, chatbot, llm, history_len, temperature],
            outputs=[msg, chatbot],
            show_progress="minimal",
        )

        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(
            respond,
            inputs=[msg, chatbot, llm, history_len, temperature],
            outputs=[msg, chatbot],
            show_progress="hidden",
        )
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
    2. 初始化数据库时间可能较长，请耐心等待。
    3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(
    server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
)
