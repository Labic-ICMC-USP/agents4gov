from browser_use import Agent
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
import os

class TaskExecutorAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    async def execute(self, task_info: Dict) -> Optional[str]:
        if not task_info.get("is_task"):
            print("⚠️ No actionable task to execute.")
            return None

        task_list: List[str] = task_info.get("task_list", [])
        if not task_list:
            print("⚠️ Task flag is True but no tasks were listed.")
            return None

        prompt = "Execute the following tasks:\n" + "\n".join(f"- {t}" for t in task_list)
        os.environ["OPENAI_API_KEY"] = self.api_key
        agent = Agent(
            task=prompt,
            llm=ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=0,
            ),
        )

        history = await agent.run()
        return history.final_result()
