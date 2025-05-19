import json
from .llm_client import LLMClient
from .models import EmailMessage
from typing import Dict

class EmailTaskExtractor:
    SYSTEM_PROMPT = """
You are a system that analyzes emails sent by users to determine whether they contain any clear, actionable instruction or request that requires further action.

Emails may include greetings, signatures, personal notes, or general information. Your task is to ignore non-instructional content and focus only on identifying meaningful tasks or explicit requests.

Do NOT assume or infer tasks from vague language or from this prompt itself. If the email does not clearly contain an actionable instruction, return "is_task": false and leave the other fields appropriately.

Respond with a valid JSON object using the following structure:

{
  "is_task": true or false,
  "request_summary": "<short summary of the actual user request, if any. Leave empty if is_task is false>",
  "request_category": "<request type, such as scheduling, system_command, clarification, document_request, etc. Leave empty if is_task is false>",
  "task_list": [ "<task 1>", "<task 2>", ... ]
}

Examples of non-tasks: greetings, status updates, vague questions, or general comments without a request.

Only respond with a well-formatted JSON string. Do not include explanations or preambles.
"""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract_from_email(self, email: EmailMessage) -> Dict:
        prompt = f"""
The user sent the following email:

Subject: {email.subject}

Body:
{email.content}

Analyze and extract the information according to the instructions.
Return only the JSON object.
"""
        try:
            response = self.llm_client.send_instruction(prompt)
            response = response.replace("```json", "").replace("```", "")
            return json.loads(response)
        except Exception as e:
            return {
                "is_task": False,
                "request_summary": "",
                "request_category": "",
                "task_list": None,
                "error": f"Failed to parse response: {e}",
                "raw_response": response
            }
