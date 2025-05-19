import poplib
import html2text
from email.parser import BytesParser
from email.policy import default
from email.utils import parsedate_to_datetime
from .models import EmailMessage
from typing import List, Optional
import ssl
import smtplib
from email.message import EmailMessage as MIMEMessage
from datetime import datetime

class POP3Client:
    def __init__(self, pop3_server: str, pop3_port: int, address: str, password: str):
        self.server = pop3_server
        self.port = pop3_port
        self.user = address
        self.password = password
        self.seen_ids = set()

    def _connect(self):
        conn = poplib.POP3_SSL(self.server, self.port)
        conn.user(self.user)
        conn.pass_(self.password)
        return conn

    def list_messages(self) -> List[dict]:
        messages = []
        conn = self._connect()
        try:
            num_messages = len(conn.list()[1])
            for i in range(num_messages, 0, -1):
                resp, lines, octets = conn.retr(i)
                msg_bytes = b"\r\n".join(lines)
                msg = BytesParser(policy=default).parsebytes(msg_bytes)
                subject = msg["subject"] or "(no subject)"
                uid = f"{i}-{subject}".strip()
                if uid not in self.seen_ids:
                    messages.append({"index": i, "id": uid, "subject": subject, "from": msg["from"]})
        finally:
            conn.quit()
        return messages

    def get_message_by_index(self, index: int) -> Optional[EmailMessage]:
        conn = self._connect()
        try:
            resp, lines, octets = conn.retr(index)
            msg_bytes = b"\r\n".join(lines)
            msg = BytesParser(policy=default).parsebytes(msg_bytes)

            subject = msg["subject"] or "(no subject)"
            sender = msg["from"] or ""
            recipient = msg["to"] or ""
            date_header = msg["date"]
            timestamp = parsedate_to_datetime(date_header) if date_header else datetime.now()

            content = ""
            html_content = ""
            attachments = []

            if msg.is_multipart():
                for part in msg.iter_parts():
                    content_type = part.get_content_type()
                    dispo = part.get_content_disposition()
                    if dispo == "attachment":
                        filename = part.get_filename()
                        if filename:
                            attachments.append(filename)
                    elif content_type == "text/plain":
                        content += part.get_content()
                    elif content_type == "text/html":
                        html_content += part.get_content()
            else:
                if msg.get_content_type() == "text/plain":
                    content = msg.get_content()
                elif msg.get_content_type() == "text/html":
                    html_content = msg.get_content()

            if not content and html_content:
                content = html2text.html2text(html_content)

            uid = f"{index}-{subject}".strip()
            self.seen_ids.add(uid)

            return EmailMessage(timestamp, sender, recipient, subject, content.strip(), attachments)
        finally:
            conn.quit()

class SMTPClient:
    def __init__(self, smtp_server: str, smtp_port: int, address: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.address = address
        self.password = password

    def send(self, to_address: str, subject: str, body: str) -> bool:
        msg = MIMEMessage()
        msg["Subject"] = subject
        msg["From"] = self.address
        msg["To"] = to_address
        msg.set_content(body)
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.address, self.password)
                server.send_message(msg)
            print(f"✅ Email sent to {to_address}")
            return True
        except Exception as e:
            print(f"❌ Failed to send email to {to_address}: {e}")
            return False
