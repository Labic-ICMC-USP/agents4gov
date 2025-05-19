from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class LLMClient:
    def __init__(self, model: str, base_url: str, api_key: str, system_prompt: str = None, temperature: float = 0.0):
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )

    def send_instruction(self, user_prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        response = self.llm.invoke(messages)
        return response.content.strip()
