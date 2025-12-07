# Agents4Gov Documentation

Welcome to the Agents4Gov documentation. This directory contains guides and tutorials to help you work with the framework.

## Project Structure

Agents4Gov is built on top of **[Open WebUI](https://github.com/open-webui/open-webui)**, a framework for running LLM-based applications with tool integration.

```
Agents4Gov/
├── tools/                 # Implemented tools for public services
├── data/                  # Example or anonymized datasets
├── docs/                  # Documentation and evaluation reports
├── config/                # Model and system configuration files
└── README.md              # Main project documentation
```

### Key Directories

- **`tools/`** - Contains all tool implementations that can be imported into Open WebUI. Each tool is a Python class that provides specific functionality to agents.
- **`data/`** - Stores datasets used for testing and evaluation, with privacy-preserving anonymization.
- **`docs/`** - Documentation, tutorials, and research reports.
- **`config/`** - Configuration files for models and system settings.

## Available Documentation

- **[How to Create a Tool](how_to_create_tool.md)** - A comprehensive step-by-step guide for creating custom tools that can be used by agents. Learn about tool structure, parameter validation, error handling, and best practices. Reference implementation: `tools/open_alex_doi.py`

## External Resources

### Open WebUI Documentation

Agents4Gov tools are designed to run within Open WebUI. For understanding the underlying framework:

- **[Open WebUI GitHub](https://github.com/open-webui/open-webui)** - Main repository and source code
- **[Open WebUI Documentation](https://docs.openwebui.com/)** - Official documentation for installation, configuration, and usage
- **[Open WebUI Tools Guide](https://docs.openwebui.com/features/plugin/tools)** - Specific documentation on how tools work within Open WebUI

### Getting Started with Open WebUI

1. Install Open WebUI: `pip install open-webui`
2. Start the server: `open-webui serve`
3. Access the interface at [http://localhost:8080](http://localhost:8080)
4. Import Agents4Gov tools through the Tools module in the UI

## Contributing

When adding new documentation:
1. Create your markdown file in this `docs/` directory
2. Update this README.md with a link to your new document
3. Use clear, descriptive titles and include practical examples
4. Follow the structure and style of existing documentation
