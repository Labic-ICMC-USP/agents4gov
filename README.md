# Agents4Gov

**Agents4Gov** is an LLM-based automated agent system that operates through email. It receives instructions sent by users, identifies actionable tasks, routes them to specialized agents, executes them, and automatically replies with the results. The system is designed to support public sector workflows and administrative services.

## 🔁 Workflow

1. Connects to an email inbox via IMAP4.
2. Analyzes each email to detect clear instructions or requests.
3. Uses an LLM to extract the task, category, and description.
4. Selects the most appropriate agent based on available descriptions (fallback: general-purpose agent).
5. Executes the task with the selected agent.
6. Generates a natural-language response summarizing the outcome.
7. Sends the response back to the original sender via SMTP.

## 📦 Project Structure

```

agents4gov/
│
├── agents/                  # Directory for specialized agents (one per Python file)
│   ├── agent\_calendar.py
│   ├── agent\_compare.py
│   └── ...
│
├── agents4gov.py              # Main operator controlling the flow
├── llm\_client.py            # Generic LLM client interface
├── email\_imap4.py            # Email receiver (IMAP)
├── email\_smtp.py            # Email sender (SMTP)
├── extractor.py             # Task extraction via LLM
├── responder.py             # User response generator
│
├── README.md
└── TODO.md

````

## 🧠 Agent Selection Logic

- Python files inside the `agents/` directory are dynamically loaded.
- Each agent module must contain a top-level docstring describing its purpose.
- The operator builds a prompt combining all available agent descriptions and the user’s task summary.
- The LLM selects the most suitable agent to handle the request.
- If no confident match is found, the general-purpose agent is used.

## 🤖 LLM Responsibilities

- Email-to-task extraction (`request_summary`, `request_category`, `task_list`)
- Agent selection based on semantic alignment between task and agent description
- Automated response generation using the user’s original message and execution result

Perfeito! Aqui está uma nova seção **“How to Use”** para o seu `README.md`, clara e objetiva, explicando como rodar o `agents4gov` do zero:

---

## 🚀 How to Use

Follow these steps to run the `agents4gov` system locally:

### 1. Clone the repository

```bash
git clone https://github.com/Labic-ICMC-USP/agents4gov.git
cd agents4gov
```

---

### 2. Install the project (editable mode)

Make sure you have Python 3.11+. Then run:

```bash
pip install -e .
```

This installs `agents4gov` and its dependencies in editable mode.

---

### 3. Configure your credentials

Create a file named `config.json` in the project root. Use the following structure:

```json
{
  "email": {
    "imap4_server": "your.imap4.server",
    "imap4_port": 993,
    "smtp_server": "your.smtp.server",
    "smtp_port": 465,
    "address": "your@email.com",
    "password": "your-email-password"
  },
  "llm": {
    "api_key": "sk-xxxxx...",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o"
  }
}
```

Then create a `.env` file pointing to it:

```env
AGENTS4GOV_CONFIG=config.json
```

---

### 4. Install Playwright browser (for the agent)

This is required for the browser automation used by the AI agent.

```bash
playwright install
```

---

### 5. Run the agent

Start the system using:

```bash
python main.py
```

It will:

* Read emails via IMAP4
* Extract user instructions
* Select the best agent
* Execute the task
* Send back a response by email


## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

This project relies on the excellent [browser-use](https://github.com/browser-use/browser-use) library for browser-based autonomous agent execution. We thank the contributors for their work in enabling intelligent automation.
