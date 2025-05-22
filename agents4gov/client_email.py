import poplib
import imaplib
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


class IMAP4Client:
    def __init__(self, imap4_server: str, imap4_port: int, address: str, password: str):
        self.server = imap4_server
        self.port = imap4_port
        self.user = address
        self.password = password
        self.seen_ids = set()

    def _connect(self):
        conn = imaplib.IMAP4_SSL(self.server, self.port)
        conn.login(self.user, self.password)
        return conn

    def list_messages(self) -> List[dict]:
        messages = []
        conn = self._connect()
        conn.select("inbox")
        status, data = conn.uid('search', None, "UNSEEN")
        email_ids = data[0].split()
        print(email_ids)
        try:
            for i in reversed(email_ids):
                email_id = i.decode()
                resp, data = conn.uid('fetch', email_id, "(RFC822)")
                msg_bytes = data[0][1]
                msg = BytesParser(policy=default).parsebytes(msg_bytes)
                subject = msg["subject"] or "(no subject)"
                uid = f"{email_id}-{subject}".strip()
                if uid not in self.seen_ids:
                    messages.append({"index": email_id, "id": uid, "subject": subject, "from": msg["from"]})
        finally:
            conn.logout()
        return messages

    def get_message_by_index(self, index: int) -> Optional[EmailMessage]:
        conn = self._connect()
        conn.select("inbox")
        try:
            resp, lines = conn.uid('fetch',index, "(RFC822)")
            msg_bytes = lines[0][1]
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
                content = "content"

            uid = f"{index}-{subject}".strip()
            self.seen_ids.add(uid)

            return EmailMessage(timestamp, sender, recipient, subject, content.strip(), attachments)
        finally:
            conn.logout()





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
