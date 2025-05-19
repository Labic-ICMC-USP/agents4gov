# Agents4Gov

**Agents4Gov** is an LLM-based automated agent system that operates through email. It receives instructions sent by users, identifies actionable tasks, routes them to specialized agents, executes them, and automatically replies with the results. The system is designed to support public sector workflows and administrative services.

## 🔁 Workflow

1. Connects to an email inbox via POP3.
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
├── email\_pop3.py            # Email receiver (POP3)
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

## 🚀 Running the System

```bash
python agents4gov.py
````

Make sure to configure your email credentials and LLM API key in `operator.py`.

## 📄 License

[MIT License](LICENSE)

