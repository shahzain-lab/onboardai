# OnboardAI: Agentic Assistant for Teams

It began in a much simpler way: I was curious about AI. I wanted to see how far these new tools, frameworks, and protocols could actually go if you stitched them together into something practical.

That curiosity eventually turned into **OnboardAI** — my attempt to build an autonomous assistant that helps teams onboard smoothly, share updates, record meetings, and encourage collaboration.

---

# What is OnboardAI?

I started learning AI with one simple curiosity: could I make these frameworks actually work in the messy reality of teamwork?

As I explored, I found dozens of orchestration frameworks:

* Some were designed to sit on top of large language models (LLMs).
* Others worked without LLMs, focusing instead on context, messaging, or workflow.

Now, putting all of them together in one project might sound a little crazy — and maybe even unnecessary from a commercial perspective. In fact, there are definitely performance trade-offs in trying to combine everything. But the point of this project was never about squeezing out maximum efficiency.

The idea here was to go full in-depth, to explore all possibilities, and to see how far I could push these frameworks when combined. Even if it isn’t the most practical setup for a production SaaS, it becomes a powerful learning source from a developer’s perspective. That’s the value I got out of it — hands-on understanding of how these pieces fit, conflict, and complement each other.

---

# OnboardAI Features

OnboardAI is my experimental project to build a Slack-based assistant that:

* Welcomes new users and walks them through onboarding steps.
* Collects standup updates from team members.
* Joins meetings (via Google Meet), records and transcribes them, then turns those into tasks.
* Nudges people to collaborate when opportunities show up.

It’s not one giant AI agent. Instead, it’s a **network of smaller agents**, each designed to handle specific responsibilities. They work in sync when needed and independently when possible.

---

# Core Idea

The central idea is simple: **make team coordination less manual.**

Instead of someone asking for updates, chasing meeting notes, or reminding new hires what to do, the assistant quietly handles it in the background.

But for me, OnboardAI is also more than just solving those workflow gaps. It’s about taking everything I’ve been learning on my AI journey — the frameworks, the protocols, the experiments — and putting it into a practical project.

That’s why this project pulls together so many different pieces, even if it feels like overkill at times. It’s not just a tool for me; it’s a way to make my own understanding concrete, while also leaving something that others can explore, pick apart, and collaborate on.

---

# Tech Stack & Architecture

This project is also my playground for exploring today’s agentic ecosystem. It’s not just about using one framework — it’s about seeing how they can fit together in practice, even if that means extra complexity.

## Backend Foundation

* **Docker**: Containerized backend for consistent environments.
* **FastAPI**: Core service handling external requests (Slack webhooks, Google events, API calls) and routing into the agent layer.

## Autogen + LangGraph

* **Autogen**: High-level routing layer, decides which workflow to trigger inside LangGraph.
* **LangGraph**: Manages multiple agent layers (transcription, summarization, task extraction, evaluation).

  * Each agent tested for **accuracy, relevance, and completeness** to ensure expected results.

## CrewAI

* Coordinates the “crew” of agents.
* Manages which agents are active, how they collaborate, and whether tasks run in parallel or sequentially.

## OpenAI Agent SDK

* Orchestration backbone for agent-tool interaction.
* Ensures agents share state, handle conversations, and pass results without conflicts.

## MCP (Model Context Protocol)

* Enables agents to share context **without everything going through an LLM**.
* Reduces unnecessary calls, keeps workflows lighter, and makes the system modular.

## LangChain & Community Tools

* Used for embeddings, retrieval, and chaining reasoning steps.
* Provides utility blocks for structured data and multi-step reasoning.

## Slack Integration

* Slack is the main interface.
* Slash commands, event subscriptions, and bot messages → flow through FastAPI → Autogen → LangGraph → agents → results back to Slack.

---

# Why Open Source?

I’m keeping this project open because I believe in learning by doing — and sharing.

* **Community-driven learning**: Others can explore the code, point out flaws, suggest improvements.
* **Adaptability**: Each team works differently; open-source allows the tool to evolve.
* **Collaboration**: The very thing the assistant is built to encourage.

---

# Future Possibilities

* Multi-agent parallel workflows.
* Integration with more orchestration frameworks as they mature.
* Tighter connection with project management dashboards.

---

# Conclusion

OnboardAI is not a polished SaaS. It’s an experiment, a learning project, and hopefully a useful assistant. It’s me testing how far this new generation of AI tools can be pushed when applied to everyday team problems.
