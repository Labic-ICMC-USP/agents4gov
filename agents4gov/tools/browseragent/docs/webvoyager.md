# WebVoyager

It introduces a new approach to building autonomous web agents capable of visually and textually understanding real-world websites to complete tasks end-to-end.


## Key Components

**WebVoyager** is an autonomous web agent that uses **Large Multimodal Models (LMMs)** to **see, understand, and interact** with real-world websites.

### Problems with Previous Web Agents
- **Text-only processing:** Earlier systems relied solely on HTML text and ignored visual layouts.  
- **Simulated environments:** Most agents were tested in simplified web simulators rather than dynamic, real websites.

WebVoyager bridges this gap by:
- Combining **visual (screenshots)** and **textual (HTML)** data.
- Operating directly on **live websites**.
- Emulating **human-like browsing behavior** to follow user instructions autonomously.

## How WebVoyager Works

WebVoyager is an **autonomous web agent** capable of browsing the **open web** in real time — understanding and interacting with webpages through both **visual** and **textual** signals to complete user-defined instructions **end-to-end**.

Given a user instruction, WebVoyager:
1. Launches a web browser.
2. Observes the current page (via screenshot and text).
3. Predicts an appropriate action.
4. Executes that action in the browser.
5. Repeats the cycle until the task is complete.

The system continuously updates its internal context with new observations and actions until it reaches a termination signal.


### Browsing Environment

WebVoyager operates on **real-world websites** using [Selenium](https://www.selenium.dev/).

- Unlike simulated environments such as *WebArena*, WebVoyager interacts directly with the **open internet**, facing realistic web challenges:
  - Floating ads  
  - Pop-up windows  
  - Dynamic and constantly changing content

This setup enables the agent to learn **robust, adaptive browsing behavior** closer to real-world user interaction.

### Interaction Formulation

WebVoyager’s browsing cycle is defined by four main components:
- **E** → Environment  
- **M** → Large Multimodal Model  
- **O** → Observation Space  
- **A** → Action Space  

At each step **t**:
1. The model receives the **context** `ct = (o1, a1, ..., ot, I)` containing previous actions and observations.  
2. It generates an **action** `at = M(ct)`, executed in the environment.  
3. The environment returns the next **observation** `ot+1 = E(ot, at)`.

The cycle continues until the agent stops or the step limit is reached.

#### Thought-Action Prompting
- Inspired by **ReAct Prompting**, WebVoyager produces both a **thought** (`st`) and an **action code** (`at`) for each step — reasoning before acting.  
- To maintain clarity, only the **three most recent observations** are kept, while all thoughts and actions are retained.

### Observation Space

The agent primarily observes **screenshots** instead of raw HTML.

#### Visual Input
- Screenshots include bounding boxes and numeric labels over interactive elements, overlaid using [GPT-4V-Act](https://github.com/ddupont808/GPT-4V-Act), a lightweight, rule-based JavaScript tool.  
- Labels and boxes help the model identify actionable elements precisely.  
- All borders and labels use **black** for clarity and consistency.

#### Textual Input
- Includes:
  - Element text content  
  - Element type  
  - `aria-label` or comment text  

#### Additional Design Choices
- All interactions occur in **a single browser tab**.  
- Execution errors trigger re-prompting with the error message included, consuming one step each retry.

---

### Action Space

WebVoyager mimics human browsing behaviors through seven key action types:

| Action | Description |
|--------|--------------|
| **Click** | Click on buttons or links. |
| **Input** | Type into text boxes after clearing old content. |
| **Scroll** | Move vertically through a page. |
| **Wait** | Pause to allow content to load. |
| **Back** | Navigate to the previous page. |
| **Jump to Search Engine** | Restart the browsing process if stuck. |
| **Answer** | Finalize the task and produce an output. |

Each action uses **numeric tags** from screenshots to reference specific webpage elements.

## Benchmark for WebVoyager

To ensure diversity, **15 representative websites** were selected to cover different aspects of daily life.

### Data Construction

The dataset was created using a **hybrid Self-Instruct + Human Verification** pipeline.

#### Seed Task Creation
- Manually sampled and rewritten tasks from **Mind2Web** (Yin et al., 2023; Deng et al., 2023).  
- Generated initial **seed tasks** for key websites such as Google Flights, Google Maps, Booking, and Wolfram Alpha.
- **Seed tasks are the initial**, manually created examples that start the data generation process. They act as high-quality prototypes or templates that guide further task generation.

#### GPT-4 Task Generation
- Used seed tasks as **in-context examples** to prompt **GPT-4 Turbo**.  
- Generated ~100 new tasks through **20 iterations**.  
- Each generated task was **manually verified and rewritten** when necessary.  
- Human-validated tasks were added back to the **Task Pool**.

#### Iterative Expansion
- Sampled new in-context examples each iteration.  
- Verified task diversity and correctness on target websites.  
- Final dataset: **40+ tasks per website**, totaling **643 tasks**.

### Annotation Process

Each task is annotated with a verified answer, categorized as either **Golden** or **Possible**.

| Label | Description |
|--------|-------------|
| **Golden** | Stable, exact answers. Comprehensive and unlikely to change in the short term. |
| **Possible** | Variable or open-ended answers, including: <br> 1- Open-ended tasks (e.g., summarization) <br> 2- Multiple valid answers <br> 3- Time-sensitive information (e.g., flight prices). |

**Statistics:**
- **22.3 %** of tasks labeled **Golden**
- **77.7 %** labeled **Possible**

This reflects both **stability** and **real-world variability** of web data.



## Experiment

### **Datasets and Metrics**

WebVoyager is evaluated across multiple benchmarks:

| Dataset | Description | Evaluation Metric |
|----------|--------------|-------------------|
| **WebVoyager Benchmark** | Custom benchmark introduced. | Task Success Rate |
| **GAIA (Mialon et al., 2023)** | 90 web browsing tasks (Level 1 & 2) with golden responses. Agent starts from Google Search since sites aren’t specified. | Task Success Rate |
| **SeeAct (Zheng et al., 2024)** | 50 online evaluation tasks; compared with SeeAct’s autonomous agent results. | Task Success Rate |

**Primary Metric:**  
> **Task Success Rate (TSR)** – measures whether the agent completes the task, without requiring optimal steps.


### Experimental Setup

### **Models Used**
| Model | Type | Description |
|--------|------|-------------|
| **GPT-4 Turbo (Vision)** | Backbone | Used as the primary model (`gpt-4-vision-preview`) for strong semantic and visual reasoning. |
| **Claude 3 Opus (Anthropic, 2024)** | Backbone | Adds diversity; used for ablation. |
| **GPT-4o (Omni, 2024)** | Backbone | Multimodal baseline with enhanced context understanding. |
| **GPT-4 (All Tools)** | Baseline | Integrates vision, browsing, code, and plugins. |
| **Text-only baseline** | Baseline | Receives only accessibility tree data (no screenshots). |

### Evaluation Method

#### **Human Evaluation**
- Human judges inspect full agent trajectories (screenshots + actions).
- Binary judgment: **Success** or **Failure**.
- 300 tasks reviewed by **3 annotators** for inter-rater reliability.

### **Automatic Evaluation**
- **GPT-4V** is used as an **auto-evaluator** (LMM-based judge).
- Input: task prompt, agent responses, and last *k* screenshots.
- Evaluator outputs binary success/failure.
- Increasing *k* (screenshots) improves consistency:

## Results

#### **Performance Highlights**
- **WebVoyager** outperforms **text-only** and **GPT-4 (All Tools)** baselines across most sites.
- Slightly weaker on **text-heavy** websites (e.g., Allrecipes, GitHub).
- Achieves **30% success** on the **SeeAct** test set (vs **26%** by SeeAct’s best agent).

| Website | GPT-4 (All Tools) | Text-only | WebVoyager | WebVoyager (GPT-4o) |
|----------|-------------------|------------|-------------|----------------------|
| **Overall** | **30.8%** | **40.1%** | **59.1%** | **55.5%** |

#### **Findings**
- **Visual + Textual modalities** are both essential:
  - Text-only fails on visually complex sites (Booking, Flights).
  - WebVoyager outperforms text-only and GPT-
    4 (All Tools) baselines by large margins in most
    website tasks, while it is slightly lower than Text-
    only on Allrecipes and similar to Text-only on
    Github, ESPN, Cambridge Dictionary and Wolfram
    Alpha. This is primarily because these websites
    are more text-heavy than others. Since WebVoy-
    ager mostly relies on web screenshots for decision-
    making, dense text might not be easily recogniz-
    able from the image.
- **Website complexity** correlates inversely with success:
  - Sites with fewer interactive elements and shorter trajectories show higher TSR.
- **Direct interaction** (vs Bing scraping) is critical for accuracy.

### Error Analysis

Manual labeling of 300 failed tasks reveals key failure modes:

| Failure Type | Description | Ratio |
|---------------|--------------|-------|
| **Navigation Stuck** | Agent fails to finish task or loops endlessly (e.g., scroll errors, vague queries). | **44.4%** |
| **Visual Grounding Issue** | Misidentifies or confuses visual elements, especially small text or nearby items. | **24.8%** |
| **Hallucination** | Produces plausible but incorrect results (e.g., partial answers, wrong inputs). | **21.8%** |
| **Prompt Misalignment** | Fails to follow task structure or prematurely answers. | **9.0%** |

---

#### **Examples**
- *Navigation Stuck:* Scrolls indefinitely due to small scroll area.
- *Visual Grounding:* Clicks wrong “Buy” button near a similar label.
- *Hallucination:* Answers with partial product info.
- *Prompt Misalignment:* Generates “Thought” but no executable action.

## Conclusion

WebVoyager is a large multimodal model (LMM)–powered web agent designed to complete real-world web tasks end-to-end by directly interacting with websites.
It combines visual and textual understanding to perform actions on web pages and significantly outperforms baseline web agents.

it introduced an automatic evaluation framework using GPT-4V to assess agent performance objectively.
This establishes WebVoyager as a strong foundation for building more capable and intelligent web assistants in the future.

### Limitations

**Incomplete Action Set**:
The agent currently lacks certain human-like actions such as dragging, due to the complexity of continuous pixel interactions.
Future improvements in visual grounding could enable this.

**Limited File Support**:
WebVoyager handles basic file types (text, PDFs) but not complex media (e.g., videos). Extending file-type support is a key area for future work.

**Risks & Safety Concerns**

Before real-world deployment, strong safety measures are required.Potential risks include:
- Downloading malicious content
- Exposing confidential data
- Sending unintended or harmful web requests
- Generating fake or automated user activity
- Strict ethical and security safeguards are needed for responsible use.