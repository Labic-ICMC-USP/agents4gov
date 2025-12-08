#!/usr/bin/env python3
"""
Local browser test for CNPq/Lattes Navigator.
Runs with visible browser to observe AI agent navigation.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_browser.py
    python test_browser.py --lattes-id 4003190744770195
"""
import os
import sys
import asyncio
import argparse

def check_deps():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    try:
        from browser_use import Agent, Browser, BrowserConfig, ChatOpenAI
        return Agent, Browser, BrowserConfig, ChatOpenAI
    except ImportError as e:
        print(f"Error: {e}")
        print("Install: pip install browser-use playwright && playwright install chromium")
        sys.exit(1)


async def run_test(lattes_id: str, name: str, headless: bool = False):
    Agent, Browser, BrowserConfig, ChatOpenAI = check_deps()
    
    print(f"\nTesting Lattes ID: {lattes_id}")
    print(f"Researcher: {name}")
    print(f"Headless: {headless}")
    print("-" * 50)
    
    browser = Browser(config=BrowserConfig(headless=headless))
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    profile_url = f"http://lattes.cnpq.br/{lattes_id}"
    
    task = f"""
TASK: Extract academic data from Brazilian Lattes CV.

DO NOT use search engines. Navigate DIRECTLY to these URLs:

STEP 1: Go to https://buscatextual.cnpq.br/buscatextual/visualizacv.do?id={lattes_id}
STEP 2: If that fails, try: {profile_url}
STEP 3: Wait for researcher name "{name}" to appear on page
STEP 4: Scroll down and look for sections (in Portuguese):
   - "Artigos completos publicados" = journal articles
   - "Projetos de pesquisa" = projects
   - "Orientacoes" = supervisions
STEP 5: Extract data from years 2020-2025 only

STEP 6: Return ONLY this JSON (no other text):
```json
{{
  "last_update": null,
  "affiliations": [],
  "publications": [{{"title": "...", "year": 2024, "type": "journal"}}],
  "projects": [{{"title": "...", "start_year": 2022}}],
  "advising": [{{"name": "...", "level": "PhD", "year": 2023}}],
  "coauthors": [],
  "warnings": []
}}
```

If page blocked or captcha, return: {{"warnings": ["captcha_blocked"], "publications": [], "projects": [], "advising": [], "affiliations": [], "coauthors": [], "last_update": null}}
"""
    
    agent = Agent(task=task, llm=llm, browser=browser)
    
    print("\nStarting browser agent...")
    print("Watch the browser window to see navigation.\n")
    
    try:
        result = await agent.run(max_steps=25)
        print("\n" + "=" * 50)
        print("RESULT:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if not headless:
            print("\nKeeping browser open for 10s...")
            await asyncio.sleep(10)
        await browser.close()


def main():
    parser = argparse.ArgumentParser(description="Test browser-use with Lattes")
    parser.add_argument("--lattes-id", default="4003190744770195", help="Lattes ID to test")
    parser.add_argument("--name", default="Ricardo Marcacini", help="Researcher name")
    parser.add_argument("--headless", action="store_true", help="Run headless (no visible browser)")
    args = parser.parse_args()
    
    asyncio.run(run_test(args.lattes_id, args.name, args.headless))


if __name__ == "__main__":
    main()

