# Lumiq — Execution-First Data Analyst

> A deterministic, execution-first AI system that computes analytical insights from data instead of hallucinating them.

---

## 2. Overview

Lumiq is a full-stack AI data analysis platform engineered to provide reliable, deterministic analytics. Rather than relying on Large Language Models (LLMs) to guess statistical properties or generate text-based approximations, Lumiq translates user queries into executable Python code, runs the computations in a sandboxed environment on actual datasets, and interprets the precise results. 

It solves the critical problem of AI hallucination in data analysis by establishing a strict architectural contract: **no data-derived insight is ever returned to the user unless it is grounded in verifiable code execution.** It is distinct from generic web interfaces by serving as a dedicated computational agent, enforcing deterministic outputs over probabilistic text generation.

---

## 3. Motivation

The core challenge with current LLM applications in data analysis is that **LLMs hallucinate.** When asked about specific datasets, standard language models often guess distributions, approximate aggregations, or confidently present fabricated metrics. Data analysts and engineers require grounded computation, not plausible text. Lumiq was built to guarantee that every insight is backed by an explicit, transparent programmatic execution against the underlying data.

---

## 4. Architecture

Lumiq operates on an **Executor-First Pipeline**. When a query enters the system, it traverses a highly structured computational routing system rather than a standard single-pass LLM prompt.

**Pipeline Flow:**
`Query → Classification → Code Generation → Execution → Result Extraction → Cognitive Analysis`

1. **Query Classification:** Determines the optimal execution mode via a 4-mode router (Executor, Hybrid, Concept, Irrelevant).
2. **Code Generation (LLM):** Synthesizes optimized, deterministic Pandas code targeting the user's specific dataset schema.
3. **Execution (Sandbox):** Evaluates the synthesized code against the dataset memory space.
4. **Result Extraction:** Captures the structural output (DataFrames, aggregations, scalars) and graphic figures (Matplotlib plots → Base64).
5. **Cognitive Interpretation:** A secondary LLM pass analyzes the *extracted factual results only*, constructing structured insights without injecting external unverified data.

---

## 5. Key Features

* **Execution-First Design:** Enforces a strict behavioral template where data insights are solely derived from executed code.
* **Deterministic Outputs:** Relies on programmatic computations, guaranteeing reproducible and mathematically accurate aggregations.
* **Dual-Mode System:** Distinctly separates the Execution Engine (code synthesis & running) from the Cognitive Engine (interpreting computed outputs).
* **Structured Response Format:** All outputs return a structured payload featuring the exact answer, key insights, detected anomalies, and relevant follow-up questions.
* **Visualization Support:** Natively supports data visualization code generation, capturing `matplotlib` objects and returning them as Base64 encoded charts directly to the user interface.
* **Dataset-Aware Reasoning:** Code generation is strictly conditioned on dataset metadata and schema context (via ChromaDB semantic retrieval) to ensure syntax validity.
* **Strict Behavioral Contract:** Engineered to hard-fail rather than guess when computations throw errors.
* **Rate-Limit Resilient LLM Client:** Employs a robust interaction client featuring an asynchronous request queue, dynamic throttling, programmatic backoff, and stateful cooldowns.
* **Multi-Key Rotation Support:** The system actively monitors key health and seamlessly rotates through multiple LLM API credentials to prevent centralized rate limit bottlenecks.
* **Supabase-Backed Persistence:** Provides durable, relational persistence for query sessions, execution logs, and dataset metadata.
* **Modern UI Elements:** Exposes functionality through rich analysis blocks rather than standard conversational chat formats.

---

## 6. Tech Stack

**Frontend:**
* Next.js / React
* Tailwind CSS

**Backend:**
* FastAPI
* Pandas
* Matplotlib

**Infrastructure:**
* Supabase
* ChromaDB
* Groq / OpenAI-compatible LLM

---

## 7. Example Workflow

**User Query:** 
*"What is the month-over-month revenue growth for the enterprise segment?"*

**Generated Code:**
```python
enterprise_df = df[df['segment'] == 'Enterprise'].copy()
enterprise_df['date'] = pd.to_datetime(enterprise_df['date'])
enterprise_df = enterprise_df.set_index('date').resample('M')['revenue'].sum()
growth = enterprise_df.pct_change() * 100
result = growth.tail(1).item()
```

**Output:**
```json
{
  "result": 12.4
}
```

**Insight:**
*"The enterprise segment achieved a 12.4% revenue growth in the most recent month. This represents an acceleration from the previous period's 8.1% growth, driven by Q4 volume increases."*

---

## 8. Rate Limit Engineering

A significant engineering challenge in executing consecutive code synthesis and cognitive evaluation loops is managing LLM provider rate limits. Exceeding these limits typically results in immediate system crashes or degraded throughput.

Lumiq mitigates this via a robust abstraction layer wrapping the downstream LLM client:
* **Asynchronous Queuing:** Inbound completion requests are buffered into a managed queue.
* **Throttling & Cooldowns:** Dynamic state controllers enforce API concurrency limits and institute programmatic cooldown periods upon receiving 429 status codes.
* **Multi-Key Rotation:** The backend actively cycles through independent LLM authorization keys, distributing the load and isolating rate limit faults.
* **Production Reliability:** This architecture ensures the analytical pipeline scales horizontally without dropping tasks under high concurrency.

---

## 9. Setup Instructions

**1. Clone the Repository**
```bash
git clone https://github.com/yourusername/Lumiq.git
cd Lumiq
```

**2. Backend Setup**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Frontend Setup**
```bash
cd ../frontend
npm install
```

---

## 10. Environment Variables

Create `.env` files in both backend and frontend environments based on the required configurations.

```ini
# Backend LLM Variables
LLM_API_KEYS="key1,key2,key3" # Comma-separated for automatic rotation
LLM_MODEL="llama-3.3-70b-versatile"
LLM_BASE_URL="https://api.groq.com/openai/v1"

# Database infrastructure
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-anon-or-service-key"
```

---

## 11. Running the Project

Start the backend server:
```bash
# In the backend directory
uvicorn main:app --reload
```

Start the frontend server:
```bash
# In the frontend directory
npm run dev
```

---

## 12. Limitations

* **Dependent on LLM Quality:** Code synthesis capabilities rely on the baseline zero-shot reasoning of the configured LLM.
* **Provider Rate Limits:** Extensive aggregations on free-tier LLM endpoints may trigger delays despite rotation logic.
* **Structured Data Dependency:** The pipeline assumes well-formed, tabular representations capable of deterministic processing.

---

## 13. Future Improvements

* **Caching Layer:** Implementing a distributed cache to skip redundant code synthesis on identical queries.
* **Enhanced Visualization Output:** Migrating to structured JSON specifications for dynamic frontend rendering over static image encoding.
* **Query Optimization:** Abstract Syntax Tree (AST) validation prior to safe execution to verify bounds.
* **Streaming Responses:** Implementing streaming pipelines for real-time computational progress rendering.

---

## 14. Conclusion

Lumiq shifts the paradigm from generic text generation back to deterministic software engineering. By ensuring an execution-first architecture, the platform guarantees that its outputs remain verifiable, computationally sound, and strictly analyst-focused.
