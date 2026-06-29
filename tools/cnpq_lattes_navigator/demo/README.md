# Demo - Local Browser Testing

Test browser-use navigation locally with visible browser.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
export OPENAI_API_KEY="sk-..."
```

## Tests

### 1. Navigation Test (Debug)

Isolates navigation issues with minimal tasks:

```bash
python test_navigation.py
```

Options:
- Test 1: Direct URL to profile
- Test 2: Search portal with ID parameter
- Test 3: Search form interaction

### 2. Full Extraction Test

Complete extraction task matching API behavior:

```bash
python test_browser.py
python test_browser.py --lattes-id 4003190744770195 --name "Ricardo Marcacini"
python test_browser.py --headless  # Run without visible browser
```

## Observed Issues

From Railway logs:
- Captcha challenges on Lattes pages
- CDP timeout errors
- Agent falling back to DuckDuckGo search

Use these tests to validate navigation paths before deploying.

