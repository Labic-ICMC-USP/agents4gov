
# Agents4Gov

**Laboratory of Computational Intelligence (LABIC – ICMC/USP)**


## Overview

**Agents4Gov** is a research and development project from **LABIC – Institute of Mathematics and Computer Sciences (ICMC/USP)** focused on building **LLM-based tools** to support and modernize **public sector services**.
The project emphasizes **local Large Language Models (LLMs)** for privacy, **data anonymization**, and the **development and evaluation of tools** for use in government and institutional environments.

---

## Installation

### 1. Install the Open WebUI Server

Agents4Gov is built on top of the **[Open WebUI](https://github.com/open-webui/open-webui)** framework, which serves as the base environment for loading and running tools.

Before starting, ensure you are using **Python 3.11** to avoid compatibility issues.

To install and run Open WebUI:

```bash
# Install Open WebUI
pip install open-webui

# Start the server
open-webui serve
```

After starting, the Open WebUI interface will be available at:
👉 **[http://localhost:8080](http://localhost:8080)**

---

### 2. Clone the Agents4Gov Repository

In the same environment, clone the Agents4Gov repository:

```bash
git clone https://github.com/icmc-usp/Agents4Gov.git
```

The `tools/` directory inside the repository contains all implemented tools.

---

### 3. Import Tools into Open WebUI

Once Open WebUI is running:

1. Access the **Tools** module in the Open WebUI interface.
2. Use the **Import Tool** option to add any of the tools from the `Agents4Gov/tools/` directory.
3. Each tool has its own documentation and configuration guide within its folder.

Example:

```bash
ls Agents4Gov/tools/
```

Each subdirectory corresponds to an individual tool that can be imported, executed, and evaluated directly within Open WebUI.

---

## Repository Structure

```
Agents4Gov/
├── tools/                 # Implemented tools for public services
├── data/                  # Example or anonymized datasets
├── docs/                  # Documentation and evaluation reports
├── config/                # Model and system configuration files
└── README.md
```

---

## Objectives

* Develop and evaluate **LLM-based tools** focused on **public sector innovation**.
* Ensure **privacy-preserving** AI development using local LLMs and anonymized data.
* Provide a **modular and extensible** framework for integrating intelligent tools into public service environments.

---

## License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.
