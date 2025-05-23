from .config import load_config
from .client_email import SMTPClient,IMAP4Client
from .llm_client import LLMClient
from .extractor import EmailTaskExtractor
from .executor import TaskExecutorAgent
from .responder import EmailResponseLLM
import asyncio

async def main():
    config = load_config()

    # client = POP3Client(
    #     pop3_server=config["email"]["pop3_server"],
    #     pop3_port=config["email"]["pop3_port"],
    #     address=config["email"]["address"],
    #     password=config["email"]["password"]
    # )
    client = IMAP4Client(
        imap4_server=config["email"]["imap4_server"],
        imap4_port=config["email"]["imap4_port"],
        address=config["email"]["address"],
        password=config["email"]["password"]
    )
    smtp = SMTPClient(
        smtp_server=config["email"]["smtp_server"],
        smtp_port=config["email"]["smtp_port"],
        address=config["email"]["address"],
        password=config["email"]["password"]
    )

    llm_extractor = LLMClient(system_prompt=EmailTaskExtractor.SYSTEM_PROMPT, **config["llm"])
    llm_responder = LLMClient(system_prompt=EmailResponseLLM.SYSTEM_PROMPT, **config["llm"])

    extractor = EmailTaskExtractor(llm_extractor)
    executor = TaskExecutorAgent(config["llm"]["api_key"], model=config["llm"]["model"])

    messages = client.list_messages()
    for msg_meta in messages:
        print(f"📨 Checking: {msg_meta['subject']}")
        full_email = client.get_message_by_index(msg_meta["index"])
        if not full_email:
            continue

        result = extractor.extract_from_email(full_email)
        print(result)

        if result.get("is_task"):
            execution_result = await executor.execute(result)
            print("🧠 Agent Result:", execution_result)

            if execution_result:
                response = EmailResponseLLM(full_email, execution_result, llm_responder)
                subject = response.generate_subject()
                body = response.generate_body()
                smtp.send(full_email.sender, subject, body)
