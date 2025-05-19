from datetime import datetime
from typing import List

class EmailMessage:
    def __init__(self, timestamp: datetime, sender: str, recipient: str, subject: str, content: str, attachments: List[str]):
        self.timestamp = timestamp
        self.sender = sender
        self.recipient = recipient
        self.subject = subject
        self.content = content
        self.attachments = attachments

    def __repr__(self):
        return f"<Email '{self.subject}' from {self.sender} to {self.recipient} at {self.timestamp}>"
