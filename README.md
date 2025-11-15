# Multifunctional LangGraph Agent

A high-performance, fully automated job-search agent designed to extract structured search criteria, generate expansive multi-strategy search queries, aggregate results from numerous sources, and validate job postings through a two-tier Playwright-based verification pipeline.

This agent was engineered for speed, resilience, and high recall.

---

#  Core Capabilities

1. Email Automation

Sends outbound emails automatically

Parses and handles email content

Can be integrated into workflows (notifications, alerts, job outreach, etc.)

2. Memory System

Persistent storage of relevant facts

Contextual recall for multi-step workflows

Enables long‑term reasoning and adaptive behavior

3. Web Search Tools

Live web search retrieval

Query analysis and parsing

Integrates search results into agent reasoning via LangGraph

4. Job Search Tools 

Scrapes job listings

Filters and ranks based on user criteria

Automates update cycles and monitoring

🕸️ Architecture

Built using LangGraph for deterministically orchestrated agent flows

Modular tool registry for plugging in new functions

Supports parallelism, retries, and robust error handling

🚀 Use Cases

Job search automation

Research assistant workflows

Daily information gathering and summarization

Automated email interactions

Memory‑augmented personal assistant tasks

### ✅ Intelligent Criteria Extraction

* LLM-driven parser that returns standardized job-title, location, experience-level, count, and salary requirements.
* Caches all previous criteria for speed and consistency.

### ✅ Multi-Strategy Query Generation

* 20+ search variants
* 100+ job-related domains
* 4 independent search strategies:

  1. Site-specific queries
  2. Broad queries
  3. In-title / in-URL pattern queries
  4. ATS platform deep scans
* Produces hundreds of unique search queries with randomized ordering for wide, non-deterministic coverage.

### ✅ Parallel DuckDuckGo Scraping

* High-volume, multi-threaded DDG fetching (20 threads).
* Automatic fallback on query failures.
* Strict normalization + deduplication pipeline for clean URL lists.

### ✅ Robust Relevance Scoring

* Lightweight heuristic scoring system.
* Removes salary pages, forum posts, aggregator spam, and profile pages.

### ✅ Two-Tier Playwright Validation

1. **Fast Validation**

   * Quick "open-status" check to ensure a posting is still live.
   * Runs with 16 workers for aggressive filtering.
2. **Full Validation**

   * Deep scan of the title, structure, and job content.
   * Ensures final results are genuinely open, relevant, and high-quality.

### ✅ Final Filtering & Output

* Deduplicated, relevance-refined job results.
* Constraints: title matching, location, experience level, salary threshold.
* Clean, structured Markdown output.

---


## **Challenges**

Building this agent required navigating several constraints—most notably that nearly all modern AI tooling, APIs, and automation platforms operate behind subscription or usage-based paywalls. Working entirely within free-tier limits introduces significant restrictions on throughput, hosting, API calls, and model selection.

To overcome these limitations, I relied on creative architectural solutions and a focus on efficient local inference. Running a local LLM became essential; without it, the agent simply could not have been built, as no online LLM API calls were possible within a zero-budget environment.

Throughout development, I encountered a wide range of issues—unexpected output formats, validation inconsistencies, and challenges optimizing both speed and accuracy. These required constant iteration, prompting refinement, and systematic debugging. This process greatly strengthened my prompt engineering skills and deepened my understanding of agent reliability and robustness.

The job search system was the most complex and rewarding component. It required continual iteration to balance:

* coverage
* performance
* speed
* relevance
* validation accuracy

The final result is a stable, high-recall job search and research agent optimized for maximum performance despite tight resource constraints.

---


## **Requirements**

* Python 3.10+
* Playwright
* ddgs
* A local LLM (e.g., LM Studio, Ollama, or compatible pipeline)
