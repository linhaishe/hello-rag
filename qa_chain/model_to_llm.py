from llm.call_llm import LLM_MODEL_DICT
from llm.call_llm import parse_llm_api_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llm.zhipuai_llm import ZhipuAILLM
from llm.spark_llm import Spark_LLM
from llm.wenxin_llm import Wenxin_LLM
import sys

sys.path.append("../llm")


def model_to_llm(
    model: str = None,
    temperature: float = 0.0,
    appid: str = None,
    api_key: str = None,
    Spark_api_secret: str = None,
    Wenxin_secret_key: str = None,
):
    """
    星火：model,temperature,appid,api_key,api_secret
    百度问心：model,temperature,api_key,api_secret
    智谱：model,temperature,api_key
    OpenAI：model,temperature,api_key
    """
    provider = next(
        (name for name, models in LLM_MODEL_DICT.items() if model in models), None
    )
    if provider == "openai":
        if api_key == None:
            api_key = parse_llm_api_key("openai")
        llm = ChatOpenAI(
            model_name=model, temperature=temperature, openai_api_key=api_key
        )
    elif provider == "wenxin":
        if api_key == None or Wenxin_secret_key == None:
            api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
        llm = Wenxin_LLM(
            model=model,
            temperature=temperature,
            api_key=api_key,
            secret_key=Wenxin_secret_key,
        )
    elif provider == "xinhuo":
        if api_key == None or appid == None and Spark_api_secret == None:
            api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
        llm = Spark_LLM(
            model=model,
            temperature=temperature,
            appid=appid,
            api_secret=Spark_api_secret,
            api_key=api_key,
        )
    elif provider == "zhipuai":
        if api_key == None:
            api_key = parse_llm_api_key("zhipuai")
        llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature=temperature)
    elif provider == "gemini":
        if api_key is None:
            api_key = parse_llm_api_key("gemini")
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )
    else:
        raise ValueError(f"model{model} not support!!!")
    return llm
