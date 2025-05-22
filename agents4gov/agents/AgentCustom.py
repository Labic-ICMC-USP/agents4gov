import os
import json
import asyncio
import hashlib
import logging
from pathlib import Path
from browser_use import Agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

class AgentCustom:
    def __init__(self, config_path: str):
        self.load_config(config_path)
        self.setup_llms()

    def load_config(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.task = config.get("task", "")
        self.message_context = config.get("message_context", "")
        self.execution_model = config.get("execution_model", "")
        self.execution_api_key = config.get("execution_api_key", "")
        self.execution_base_url = config.get("execution_base_url", "")
        self.planning_model = config.get("planning_model", "")
        self.planning_api_key = config.get("planning_api_key", "")
        self.planning_base_url = config.get("planning_base_url", "")
        self.planner_interval = config.get("planner_interval", 1)
        self.output_directory = config.get("output_directory", "results")

    def setup_llms(self):
        os.environ["OPENAI_API_KEY"] = self.planning_api_key
        os.environ["OPENAI_BASE_URL"] = self.planning_base_url

        self.execution_llm = ChatOpenAI(
            model=self.execution_model,
            openai_api_key=self.execution_api_key,
            api_key=self.execution_api_key,
            base_url=self.execution_base_url
        )

        self.planner_llm = ChatOpenAI(
            model=self.planning_model,
            openai_api_key=self.planning_api_key,
            api_key=self.planning_api_key,
            base_url=self.planning_base_url
        )

    async def run(self):
        agent = Agent(
            task=self.task,
            message_context=self.message_context,
            llm=self.execution_llm,
            planner_llm=self.planner_llm,
            planner_interval=self.planner_interval,
            use_vision=False,
            use_vision_for_planner=False
        )
        result = await agent.run()
        await self.save_output(result)
        return result


    async def save_output(self, result):
        os.makedirs(self.output_directory, exist_ok=True)
        hash_id = hashlib.md5(self.task.encode('utf-8')).hexdigest()
        output_file = Path(self.output_directory) / f"{hash_id}.json"

        # Convertendo cada item do resultado para dict se possível
        def to_dict_safe(item):
            if hasattr(item, 'to_dict'):
                return item.to_dict()
            elif hasattr(item, '__dict__'):
                return vars(item)
            else:
                return str(item)

        result_list = [to_dict_safe(r) for r in result]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_list, f, indent=2, ensure_ascii=False)

        print(f"[✓] Results saved: {output_file}")



class FileLoggingHandler(logging.Handler):
    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.log_file_path = file_path
        self.log_file = self.log_file_path.open(mode="a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        self.log_file.write(log_entry + "\n")
        self.log_file.flush()

    def close(self) -> None:
        if self.log_file:
            self.log_file.close()
        super().close()


async def main():
    agent_runner = AgentCustom(config_path="agent_config.json")
    result = await agent_runner.run()
    print("=== FINAL RESULT ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
