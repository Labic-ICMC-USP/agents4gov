# MiniWoB++ Overview

MiniWoB++ is a benchmark of small **synthetic web tasks** designed to test a browser agent’s **basic interaction skills** (clicking, typing, dragging). These low-level skills are fundamental for tackling more complex, real-world web tasks.

### 1. Type

MiniWoB++ uses **synthetic mini webpages** built with fully controlled **HTML/CSS/JS**.  
This allows complete control over layout, difficulty, timing, and randomness.

### 2. Observation Space

Agents “observe” the webpage through two main modalities:

#### **DOM-based observations**
- Structured HTML elements  
- Attributes, classes, text content  
- Tree representation of the page

#### **Pixel-based observations**
- Screenshot of the rendered page  
- Useful for vision-based agents


### 3. Action Space

The agent interacts through **low-level UI actions**, similar to real browser events:

- `click(x, y)`
- `type(text)`
- `press_key(key)`
- `drag(start → end)`
- `select_option`
- `scroll`

These actions form the agent’s **interaction vocabulary**.

### 4. Tasks

MiniWoB++ contains **over 100 tasks**, grouped into two categories:

#### **A. Low-level tasks**
Simple, atomic interactions:
- click-button  
- click-checkbox  
- enter-text  
- drag-item  
- focus-text  
- scroll  

#### **B. Higher-level synthetic tasks**
More complex but still controlled:
- choose-date  
- use-autocomplete  
- find-matching-item  
- multi-step form filling  
- small “flight booking” task  

### 5. Metrics

Each task outputs a score, typically based on:

- **Task completion** (success / fail or 0–1 reward)
- **Time taken**
- **Number of mistakes**
- **Correctness of typed inputs**

These metrics help evaluate fine-grained interaction performance.

# WebArena Overview

WebArena is a **realistic web environment** for evaluating browser agents.   Unlike MiniWoB++, it does **not** use synthetic pages.

Instead, it provides full, interactive, self-hosted websites — realistic but safely contained within a closed environment.

### 1. Websites Included

WebArena simulates **four functional web applications**, each representing a different real-world domain:

- **Forum** (similar to Reddit or Discourse)  
- **E-commerce platform** (similar to Amazon)  
- **Wiki** (similar to Wikipedia)  
- **Social media / blogging platform**

Agents must complete tasks such as:

- create a post  
- reply to users  
- search for products  
- add items to cart  
- edit wiki pages  
- navigate categories  
- manage account settings  

These tasks are **far more complex** than the small, controlled tasks of MiniWoB++.

### 2. Observation Space

Agents receive rich and realistic observations:

- **DOM tree** (full HTML structure)
- **Screenshots** of the rendered page
- **URL and browser metadata**
- **Accessibility tree** (in some setups)

This resembles MiniWoB++ but on **much larger and dynamic pages**.

### 3. Action Space

Agents interact through realistic browser actions:

- click  
- type  
- select  
- scroll  
- navigate URLs  
- fill and submit forms  
- interact with search bars  
- multi-step navigation across pages  

WebArena essentially exposes a **real browser environment**.

### 4. Metrics

Tasks are evaluated based on:

- **Success / failure**
- Whether the **final webpage state** matches the goal
- **Partial credit** for progress toward multi-step tasks
- Scores aggregated across multiple tasks

This mirrors how a human would be evaluated when completing tasks on real websites.

# BrowserGym Overview

BrowserGym is **not a benchmark** — it is a **framework** for training, evaluating, and standardizing browser agents. 

Think of it as: **“OpenAI Gym / Gymnasium, but for web agents.”**

It provides the infrastructure needed so researchers can plug in many different environments (e.g., MiniWoB++, WebArena) without reinventing observation formats, actions, or reward loops.

### 1. Type

BrowserGym is **a unified framework** that supports **multiple web environments**, both synthetic and realistic.

Examples of environments it can load:

- **MiniWoB++** (synthetic tasks)
- **WebArena** (realistic websites)
- Custom local websites
- HTML task collections
- BrowserRL environments
- Human-demonstration-based tasks

### 2. Observation Space

BrowserGym standardizes what an agent receives as input, ensuring consistency across environments:

- **DOM tree**
- **Screenshots**
- **Accessibility Tree**
- **Browser metadata**
- **URL**
- **Element bounding boxes**
- **Extracted text content**

Agents get a **structured and uniform API**, regardless of which environment is loaded.

### 3. Action Space

Just like OpenAI Gym standardizes actions, BrowserGym defines a consistent browser-interaction API:

- `click(x, y)`
- `type(text)`
- `focus(element)`
- `keypress`
- `scroll`
- `select_option`
- `navigate(url)`
- Interact with browser tabs

This makes agents **portable**:  
Train in one environment, test in another with minimal changes.

### 4. Tasks

BrowserGym **does not define tasks**.  Instead, tasks are loaded from whichever benchmark or dataset the user selects:

- MiniWoB++ tasks  
- WebArena tasks  
- Custom scripted tasks  
- Human demonstration workflows  
- Recorded trajectories  

## 5. Metrics

BrowserGym also **does not define metrics**.  
It simply forwards metrics from each environment:

- MiniWoB++ reward signals  
- WebArena success criteria  
- Custom environment scoring  

Metrics are determined by the underlying benchmark, not BrowserGym.


# Comparison Table

| Benchmark     | Type (Synthetic / Real Web) | Observation Space | Action Space | Tasks & Metrics | Setup |
|---------------|-----------------------------|-------------------|--------------|-----------------|-------|
| **MiniWoB++** | Synthetic mini web pages | DOM, element attributes, text, screenshots | Low-level browser actions (click, type, select, focus); sometimes coordinate-based | Short, single-step or few-step tasks (click a button, fill a field, choose an item). Metrics: task success, reward, completion time | Lightweight, local, deterministic HTML tasks; trivial to run and reset |
| **WebArena**  | Realistic, closed-world websites (e-commerce, forums, dashboards, tools) | Full DOM, page render, text, rich element metadata | Full browser interaction (click, type, scroll, navigate, multi-step workflows) | Long-horizon, realistic tasks requiring planning (shopping, posting, searching, editing). Metrics: success, sub-goals, task score | Heavy setup; Docker environment hosting multiple real-like websites |
| **BrowserGym** | Framework hosting multiple benchmarks including MiniWoB++, WebArena, and others | Depends on selected environment; supports DOM, screenshots, accessibility tree, text | Unified, standardized browser action API across all supported benchmarks | Not a benchmark itself—aggregates many. Metrics depend on each integrated benchmark but share unified API, logging, and evaluation | Install BrowserGym; load any integrated environment; provides standardized APIs and wrappers