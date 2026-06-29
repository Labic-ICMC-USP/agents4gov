# OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use"

It presents a comprehensive survey on **OS Agents**, a class of advanced AI assistants powered by **Multimodal Large Language Models (MLLMs)**. The core idea is to move beyond domain-specific AI to create general-purpose agents.

## Key Components

OS Agents are based on several key components
and necessitate some core capabilities discussed in
the following.

1.  **Understanding:** The ability to perceive and analyze the current state of the OS environment. This involves processing GUI screenshots, extracting relevant information, and understanding the user's goal. Techniques include visual description, semantic description, and action-oriented description, often leveraging MLLMs for deep comprehension of visual and textual elements.
2.  **Planning:** The process of breaking down a complex, high-level user request into a sequence of executable, low-level actions. This often involves hierarchical planning, where MLLMs generate a high-level plan which is then refined into specific steps. Iterative planning, where the plan is adjusted after each action, is also a common approach.
3.  **Grounding (Action):** The capability to translate the planned actions into concrete interactions with the OS environment. This is crucial for bridging the gap between the agent's abstract plan and the physical execution on the screen. It involves identifying the correct target element (e.g., a button or text field) and performing the corresponding action (e.g., click, type, scroll).





## Construction of OS Agents

Constructing OS Agents involves developing **foundation models** and **agent frameworks** that can perceive, understand, and interact with graphical user interfaces (GUIs). These models integrate **language**, **vision**, and **action** understanding through a multi-stage training pipeline composed of:

- Architecture design  
- Pre-training  
- Supervised Fine-Tuning (SFT)  
- Reinforcement Learning (RL)


### Foundation Model

Foundation models for OS Agents combine multimodal architectures with multi-phase training to bridge the gap between natural language understanding and GUI interaction.


#### Architecture

Four common architectural approaches are used in current OS Agent research:

1. **Existing LLMs**  
   - Utilize open-source large language models (LLMs) capable of processing textual instructions and HTML structure.  


2. **Existing MLLMs**  
   - Use multimodal large language models (MLLMs) that process both text and visual inputs, enabling direct GUI comprehension. 

3. **Concatenated MLLMs**  
   - Combine a separate vision encoder and language model via adapters or cross-attention modules.  

4. **Modified MLLMs**  
   - Extend standard MLLMs to handle **high-resolution GUI inputs**.  


### Pre-training

Strengthen the model’s understanding of GUIs and the correlation between visual and textual modalities.

#### Data Sources
1. **Public Data:** Used for large-scale pre-training.
2. **Synthetic Data:** Complements public data to increase coverage and diversity.

#### Tasks
- **Screen Grounding:** Extract 2D coordinates or bounding boxes for interface elements from text prompts.  
- **Screen Understanding:** Capture semantic meaning and structure of entire GUI screens.  
- **Optical Character Recognition (OCR):** Identify text within GUI components (e.g., using Paddle-OCR).

### Supervised Fine-Tuning (SFT)

Adapt pre-trained models for specific GUI navigation and grounding tasks.

#### Data Collection Techniques
1. **Rule-Based Data Synthesis:** Use automated algorithms such as BFS to explore app functions and generate trajectories.  
2. **Model-Based Data Synthesis:** Employ (M)LLMs (e.g., GPT-4V) to produce annotated samples for GUI grounding or summarization tasks (Zhang et al., 2024f).  
3. **Model-Based Data Augmentation:** Generate **Chain-of-Action-Thought (CoAT)** data, containing screen descriptions, reasoning steps, and predicted actions to boost navigation and reasoning capabilities.

### Reinforcement Learning (RL)

Align OS Agents’ behavior with task objectives through reward-driven learning, enabling them to plan, act, and adapt dynamically within GUIs.

Reinforcement learning enables OS Agents to:
- Learn adaptive strategies for complex GUI navigation tasks.  
- Align multimodal perception with real-world action outcomes.  
- Integrate hierarchical planning and in-context reasoning for better autonomy.

## OS Agent Framework


An **OS Agent framework** defines how an agent perceives, plans, remembers, and acts within an operating system environment.  
Each component contributes to creating agents capable of autonomously navigating, understanding, and operating GUIs in dynamic, multi-step tasks.


### Perception

**Perception** enables the agent to observe its environment and extract relevant information to support planning, action, and memory.

#### Input Modalities

1. **Textual Description of the OS**  
   - Early systems relied on text-based representations of the environment (e.g., HTML, DOM, or accessibility trees) because LLMs could not process visual inputs.  

2. **GUI Screenshot Perception**  
   - With the rise of MLLMs, agents can now process **visual screenshots**, aligning perception with human-like understanding. 

#### Description Techniques
- **Visual Descriptions:** Use visual cues (e.g., layout, color, icons) to improve grounding.
- **Semantic Descriptions:** Incorporate textual meaning of elements.
- **Dual Descriptions:** Combine both visual and semantic information for more robust understanding.

### Planning

**Planning** defines how an agent generates and executes a sequence of actions to achieve a goal. It enables task decomposition and dynamic decision-making.

#### Two Planning Approaches

1. **Global Planning**  
   - Generates a one-time plan that the agent executes without modification.  
   - Based on **Chain-of-Thought (CoT)** reasoning (Wei et al., 2023), allowing models to break complex tasks into structured steps. 

2. **Iterative Planning**  
   - Continuously adapts plans based on feedback and environmental changes.  
   - Builds on **ReAct** (Yao et al., 2023), combining reasoning with the results of actions.  
   - Example systems include **Auto-GUI** (Zhang & Zhang, 2023), which iteratively refines plans using past actions and CoT reasoning.


### Memory

**Memory** allows OS Agents to retain information, adapt to context, and optimize decision-making over time.  It is essential for long-term learning, adaptation, and error correction.

#### Memory Types

1. **Internal Memory**: Stores transient data such as past actions, screenshots, and states.  

2. **External Memory**: Provides long-term contextual or domain knowledge from databases, tools, or online sources.  

#### Memory Optimization Strategies

1. **Management:**  Abstract and condense redundant data, retaining only relevant insights.

2. **Growth Experience:**  Learn from prior task attempts by revisiting successful and failed steps.

3. **Experience Retrieval:** Retrieve and reuse knowledge from similar past scenarios to reduce redundant actions.W

### Action

**Action** defines how OS Agents interact with digital environments, including computers, mobile devices, and web interfaces.

#### Action Categories

1. **Input Operations**: Fundamental interactions via **keyboard**, **mouse**, or **touch** input. 

2. **Navigation Operations**: Allow agents to move across applications, interfaces, or websites. Include both **basic platform navigation** and **web-specific traversal**.

3. **Extended Operations**: Provide advanced, dynamic capabilities beyond basic input and navigation:  
     - **Code Execution:** Execute scripts or commands to extend control.
     - **API Integration:** Connect to third-party tools or services for specialized functionalities.

## Evaluation of OS Agents

how OS Agents are evaluated through standardized **metrics** and **benchmarks** to measure accuracy, efficiency, and adaptability across platforms.

### Evaluation Metrics

Two main levels of evaluation are used:

- **Step-Level Evaluation:**  
  Analyzes each individual action for correctness and grounding accuracy — how well the agent identifies and interacts with the right interface element.

- **Task-Level Evaluation:**  
  Measures overall task success and efficiency.  
  - **Success Rate (SR):** Percentage of fully completed tasks.  
  - **Step Ratio:** Compares the agent’s actions to an optimal (human) baseline — lower is better.

### Evaluation Benchmarks

Benchmarks test OS Agents in realistic digital environments using diverse **platforms** and **task types**.

### Platforms
- **Computer:** Complex, multi-application desktop systems.  
- **Phone:** Mobile GUIs requiring precise touch and gesture control.  
- **Browser:** Web-based environments with dynamic content.  

Some benchmarks combine platforms to test **cross-system transferability**.

### Task Types
1. **GUI Grounding:** Match language instructions to visual interface elements.  
2. **Information Retrieval:** Navigate and extract data from GUIs.  
3. **Agentic Tasks:** Execute full, goal-driven workflows autonomously.

## Challenges and Future Directions

1.  **Generalization and Robustness:** Agents struggle to generalize to unseen interfaces and maintain robustness against minor UI changes.
2.  **Long-Horizon Planning:** Current agents often fail on tasks requiring many steps or complex, multi-stage reasoning.
3.  **Efficiency and Cost:** The reliance on large MLLMs makes inference slow and computationally expensive.
4.  **Multi-Agent Collaboration:** Exploring frameworks where multiple specialized agents can collaborate to solve complex tasks is a promising direction.
5.  **Ethical and Safety Concerns:** As agents gain more control over user environments, ensuring their safety, security, and adherence to ethical guidelines becomes paramount.
