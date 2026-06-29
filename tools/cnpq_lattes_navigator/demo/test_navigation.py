#!/usr/bin/env python3
"""
Simple navigation test - just checks if Lattes page loads.
No extraction, minimal task to debug navigation issues.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_navigation.py
"""
import os
import sys
import asyncio

def check_deps():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    try:
        from browser_use import Agent, Browser, BrowserConfig, ChatOpenAI
        return Agent, Browser, BrowserConfig, ChatOpenAI
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


async def test_direct_url():
    """Test 1: Direct URL navigation"""
    Agent, Browser, BrowserConfig, ChatOpenAI = check_deps()
    
    print("\n" + "=" * 50)
    print("TEST 1: Direct URL Navigation")
    print("=" * 50)
    
    browser = Browser(config=BrowserConfig(headless=False))
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    task = """
Go directly to http://lattes.cnpq.br/4003190744770195
Wait for the page to load.
Tell me what you see on the page.
Return: {"success": true/false, "page_title": "...", "error": null}
"""
    
    agent = Agent(task=task, llm=llm, browser=browser)
    
    try:
        result = await agent.run(max_steps=10)
        print(f"Result: {result}")
    finally:
        await asyncio.sleep(5)
        await browser.close()


async def test_search_portal():
    """Test 2: Search portal navigation"""
    Agent, Browser, BrowserConfig, ChatOpenAI = check_deps()
    
    print("\n" + "=" * 50)
    print("TEST 2: Search Portal Navigation")
    print("=" * 50)
    
    browser = Browser(config=BrowserConfig(headless=False))
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    task = """
Go directly to https://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=4003190744770195
Wait for the page to load.
Tell me what you see on the page.
Return: {"success": true/false, "page_title": "...", "error": null}
"""
    
    agent = Agent(task=task, llm=llm, browser=browser)
    
    try:
        result = await agent.run(max_steps=10)
        print(f"Result: {result}")
    finally:
        await asyncio.sleep(5)
        await browser.close()


async def test_search_form():
    """Test 3: Use search form"""
    Agent, Browser, BrowserConfig, ChatOpenAI = check_deps()
    
    print("\n" + "=" * 50)
    print("TEST 3: Search Form")
    print("=" * 50)
    
    browser = Browser(config=BrowserConfig(headless=False))
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    task = """
1. Go to https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar
2. Wait for search form to load
3. Find input field for "Nome" (name)
4. Type "Ricardo Marcacini"
5. Click search button
6. Wait for results
7. Click first result
8. Tell me what you see
Return: {"success": true/false, "found_profile": true/false, "error": null}
"""
    
    agent = Agent(task=task, llm=llm, browser=browser)
    
    try:
        result = await agent.run(max_steps=15)
        print(f"Result: {result}")
    finally:
        await asyncio.sleep(5)
        await browser.close()


async def main():
    print("Lattes Navigation Tests")
    print("Watch the browser window to see what happens.\n")
    
    tests = [
        ("Direct URL", test_direct_url),
        ("Search Portal", test_search_portal),
        ("Search Form", test_search_form),
    ]
    
    print("Available tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    
    choice = input("\nRun test (1-3, or 'all'): ").strip()
    
    if choice == "all":
        for name, test in tests:
            await test()
            print("\n")
    elif choice in ["1", "2", "3"]:
        await tests[int(choice) - 1][1]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())

