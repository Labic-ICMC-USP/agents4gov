from .llm_client import LLMClient
from .models import EmailMessage

class EmailResponseLLM:
    SYSTEM_PROMPT = """
You are an assistant that replies to user emails after an automated task has been executed.

You must read the original user email (subject and body), and the agent's execution result. 
Your goal is to generate a **natural, polite, and helpful response** to the user. It should follow the same language used in the user's email (e.g., Portuguese, English, etc.).

Be concise and respectful. Include a short acknowledgment, summarize what was done, and embed the agent result clearly.

Never invent content. Only use what's in the original email and the agent result.

Respond only with the final email body to be sent back to the user.
"""
    def __init__(self, original_email: EmailMessage, agent_result: str, llm_client: LLMClient):
        self.original_email = original_email
        self.agent_result = agent_result
        self.llm_client = llm_client

    def generate_subject(self) -> str:
        return f"Re: {self.original_email.subject}"

    def generate_body(self) -> str:
        prompt = f"""
Original Subject: {self.original_email.subject}

Original Email:
{self.original_email.content}

Agent Result:
{self.agent_result}

Generate the email response below:
"""
        try:
            return self.llm_client.send_instruction(prompt)
        except Exception as e:
            return f"""Hello,

This is an automated response regarding your recent request. Our system has processed it successfully.

Agent Result:
{self.agent_result}

Best regards,
Agents4Gov
"""
