## Objective

Create a **Tool** for **Agents4Gov (LABIC – ICMC/USP)** that uses **browser-use** to navigate **public** CNPq/Lattes pages, starting from the **official search portal**:

**Start URL:** https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar

Given a list of **names** and **Lattes IDs**, the tool will:
1) **Detect potential Conflicts of Interest (COI)** between the listed researchers.  
2) **Summarize academic production over the last 5 years** per researcher.

---

## Scope & Constraints

- **Data sources:** Only public CNPq/Lattes pages reachable from the start URL above.  
---

## Inputs

- **Researchers (list):**  
  - `name` (string)  
  - `lattes_id` (string; as seen in the public Lattes URL)  
- **Window:** Rolling **last 5 years** (relative to execution date), configurable.  
- **COI configuration (optional):** thresholds and toggles for each rule (see below).  
---

## Conflict of Interest (COI) — Rules & Determination

The tool must evaluate **pairwise COI** across all input researchers using **only publicly available information**.  
A COI flag is raised when **any** activated rule is satisfied. Each hit must include **why** it was triggered and **evidence URLs**.

### Time Window
- Default: **last 5 calendar years** (configurable).

### Core Rules (activate via config; default = ON)
1. **Co-authorship (R1)**  
   - Condition: At least **1 co-authored** item (journal, conference, chapter, book, patent, software, technical report) within the window.  
   - Evidence: Publication entry (title, year, venue) on both profiles and/or shared coauthor list.

2. **Advisor–Advisee Relationship (R2)**  
   - Condition: One researcher listed as **advisor/supervisor** of the other’s **Master/PhD/Postdoc** within the window (concluded or ongoing).  
   - Evidence: Advising/supervision sections (names, titles, years).

3. **Institutional Overlap (R3)**  
   - Condition: **Same department or graduate program** affiliation **concurrently** within the window.  
   - Evidence: Affiliation fields (institution, unit/program, time markers).  
   - Configurable detail: Require **same program** or accept **same institution** as sufficient.

4. **Project Team Overlap (R4)**  
   - Condition: Participation in the **same funded project** (research/project section) within the window.  
   - Evidence: Project title, sponsor, role, and years as listed publicly.

5. **Committee/Board/Event Overlap (R5)**  
   - Condition: Publicly listed service on the **same committee/board/event organization** within the window (when available).  
   - Evidence: Activities/Services section with event/committee name and year.

6. **Frequent Co-Authorship (R6, stronger signal)**  
   - Condition: **≥ 3** co-authored items within the window.  
   - Evidence: Publication list corroborating repeated collaboration.

7. **Strong Institutional Proximity (R8)**  
   - Condition: **Same lab/group** explicitly named in both profiles within the window.  
   - Evidence: Group/lab names in affiliations or projects.

> **Note:** Disambiguation must be conservative. If names/venues are ambiguous, flag with **low confidence** and include a warning.

---

## Outputs

### Per Researcher
- `person`: `{ name, lattes_id, profile_url, last_update (if available) }`
- `production_5y`:
  - `publications`: counts by type; top items (title, year, venue)
  - `projects`: active/ended (title, role, sponsor, years)
  - `advising`: MS/PhD/Postdoc concluded and ongoing
  - `activities`: committee/board/event roles (if public)
  - `affiliations_5y`: institutions/programs detected
- `coauthors_5y`: unique coauthors (name, count)
- `warnings`: rate limit, missing sections, parsing ambiguity
- `evidence`: list of supporting URLs/snippets

### Pairwise COI Matrix
- `pairs`: `[ { a_lattes_id, b_lattes_id, rules_triggered: [R1, R3, ...], confidence: "high|medium|low", evidence_urls: [...] } ]`

### Summary Text (LLM-assisted if enabled)
- Short, neutral summary of COI findings and 5-year production highlights.

---

## Functional Requirements

1. **Navigation & Parsing (browser-use)**
   - Start at: `https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar`
   - Search by `name` or go directly via `lattes_id` URL when available.
   - Visit each **public profile**; extract publications, projects, advising, affiliations, activities/services.
   - Record **evidence URLs** and minimal text snippets for each extracted item.

2. **Time Filtering & Normalization**
   - Filter items to last 5 years; handle year parsing and ranges.
   - Normalize names (Unicode/case), venues, and roles; deduplicate by DOI or title+year.

3. **COI Evaluation**
   - Apply rules R1–R7
   - Assign **confidence** levels (e.g., exact match = high; fuzzy/ambiguous = low).
   - Attach **why** + **evidence URLs** to each rule hit.
---

## Expected Behavior (User Flow)

1. User opens **Open WebUI → Tools → CNPq/Lattes Navigator (COI + 5Y Summary)**.  
2. Provides a list of `{ name, lattes_id }` and optional COI config (rules ON/OFF, window).  
3. Tool navigates from the **start URL**, finds profiles, extracts public data.  
4. Tool returns:
   - JSON (per-researcher results + pairwise COI matrix)  
   - Short summary text (LLM-assisted if enabled)  
   - Action log for auditing

---

## Deliverables

- [ ] Folder: `tools/cnpq_lattes_navigator/`  
  - [ ] `README.md` — usage, COI rules, limitations, ethics/compliance  
  - [ ] `requirements.txt` — declared dependencies  
  - [ ] `main.py` — orchestration: navigation, parsing, COI rules, outputs  
  - [ ] `schema.json` — output schema (per-person + pairs)  
  - [ ] `examples/` — sample input and anonymized output JSON  
- [ ] Update `docs/README.md` to reference this tool

---

## Acceptance Criteria

- [ ] Starts navigation from the official search URL and reaches public Lattes profiles.  
- [ ] Accepts list of `{ name, lattes_id }`.  
- [ ] Extracts and summarizes **last 5 years** of production per researcher.  
- [ ] Applies COI rules (R1–R6; optional R7–R8) and returns pairwise findings with **evidence URLs** and **confidence**.  
- [ ] Returns validated JSON per `schema.json` + short human summary.  
- [ ] Implements rate limiting, retry/backoff, and transparent action logs.  
- [ ] Runs inside Open WebUI Tools (importable, configurable, runnable).

---