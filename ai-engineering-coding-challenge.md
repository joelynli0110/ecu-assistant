# Code Challenge: The "ME Engineering Assistant" Agent

## Objective

Design, build, and package a multi-tool AI agent that can assist our engineers by answering questions about different Electronic Control Unit (ECU) specifications. The agent must be able to intelligently select the appropriate documentation to consult based on the user's query.

This challenge will assess your ability to design and implement robust RAG and agentic workflows using MLOps best practices.

## Scenario

An engineer at "ME" needs to quickly cross-reference information from two different product lines: the **ECU-700 Series** (older, documented in Markdown files) and the **ECU-800 Series** (newer, documented in Markdown files).

The goal is to build an agent that, when asked a question like *"What is the maximum operating temperature for the ECU-800b?"* or *"Compare the CAN bus speed of the ECU-750 and the ECU-850"*, can autonomously retrieve and synthesize the necessary information to provide a coherent answer.

## Technical Stack & Constraints

* **Language:** Python
* **Core Logic:** LangChain & LangGraph
* **Vector Storage:** An in-memory vector store (e.g., FAISS, ChromaDB)
* **Models:** Use any publicly available LLM and Embedding models (e.g., OpenAI, Anthropic, local models via Ollama)
* **Packaging:** The agent should be logged to **MLflow** for model versioning and tracking
* **Strategy:** Adhere to "everything-as-code" and "python-package-first" principles. The final solution should not be a monolithic notebook, but a proper installable python package

---

## Challenge Structure & Timeline

**Total Time:** 8-10 hours over **10 days**

This challenge uses a **tiered evaluation approach** that allows you to demonstrate competency across different aspects of senior AI engineering while providing multiple paths to success.

### **Tier 1: Core AI/ML Engineering (60% of evaluation)**
*Essential skills - must be demonstrated for passing grade*

**Requirements:**
- Functional multi-source RAG system with ECU-700/800 document retrieval
- Working LangGraph agent with intelligent query routing
- Basic MLflow model logging with predict() method
- Clear architectural documentation and design rationale

**Success Criteria:**
- Agent correctly answers 8/10 predefined test queries
- Response time <10 seconds per query
- Code passes basic quality checks (pylint score > 85%)

**Recommended Time:** 5-6 hours

### **Tier 2: Production & MLOps Excellence (30% of evaluation)**
*Production readiness - distinguishes senior from mid-level candidates*

**Requirements:**
- Proper Python package structure with setup.py/pyproject.toml
- MLflow model logging with versioning and metadata tracking
- Comprehensive testing and validation strategy (documented)
- Performance monitoring and error handling

**Success Criteria:**
- Package is installable and can be imported as a module
- MLflow model is properly logged with all dependencies
- Error handling covers common failure modes
- Unit tests cover key functionality

**Recommended Time:** 2-3 hours

### **Tier 3: Innovation & Leadership (10% of evaluation)**
*Advanced capabilities demonstrating technical leadership*

**Requirements (Choose 1-2):**
- Implemented evaluation framework with MLflow evaluation
- Human-in-the-loop mechanisms for low-confidence scenarios
- Scalability strategy with detailed implementation plan
- Advanced agent behaviors (multi-step reasoning, tool use)

**Success Criteria:**
- Custom evaluation metrics implemented and logged
- Advanced features demonstrate clear business value
- Scalability analysis includes concrete implementation steps

**Recommended Time:** 1-2 hours

---

## Core Deliverables

Your final submission must be a well-structured Python package that includes:

### **1. Custom MLflow Model**
The complete LangGraph agent, packaged as a custom `mlflow.pyfunc` model with a `predict` method that accepts user queries and returns agent responses.

### **2. Python Package**
A proper Python package structure with:
- `setup.py` or `pyproject.toml` for installation
- Modular code organization (not a monolithic script)
- Clear separation of concerns (data loading, RAG logic, agent logic, etc.)

### **3. Comprehensive Documentation**
A project `README.md` that serves as the single source of truth, including:

- **Architectural Design:** High-level design and key decisions (chunking strategy, agent graph structure, etc.)
- **Setup & Installation:** Clear instructions for installing dependencies and running the agent
- **Usage Examples:** How to run the agent and example queries
- **Testing & Validation Strategy:** Conceptual framework for production validation, including:
  - Proposed evaluation metrics and automated testing approaches
  - Domain expertise validation methods (e.g., using MLflow evaluation with subject matter expert judgments, golden datasets, or domain-specific benchmarks)
  - Strategies for continuous validation and monitoring of agent performance in production
- **Limitations & Future Work:** Discussion of approach limitations and potential improvements

---

## What We Provide

### **Sample Data**
- Small dataset of sample documents (`ECU-700-manual.md`, `ECU-800a.md`, `ECU-800b.md`)
- Test queries to validate your implementation

### **Technical Fallback Options**
- **Vector Store Issues:** Since documents are relatively small, you can implement a fallback strategy that passes document content directly as context to the LLM
- **Alternative Frameworks:** If LangGraph proves challenging, alternative agent frameworks are allowed with proper justification
- **Model Selection:** You may use any LLM provider (OpenAI, Anthropic, local models, etc.) - just document your choice and rationale

---

## Evaluation Criteria

### **Technical Excellence**
1. **Functionality & Robustness:** Does the agent correctly answer different types of queries? How does it handle edge cases?
2. **Code Quality:** Is the solution well-designed, modular, and maintainable? We value clean code and sound software engineering principles.

### **Production Readiness**
3. **MLOps Integration:** How effectively have you used MLflow for model versioning and tracking? Your ability to deliver a deployable, versioned artifact is a key indicator of seniority.
4. **Package Quality:** Does your solution demonstrate production-ready practices (proper dependency management, error handling, testing)?

### **Strategic Thinking**
5. **Architectural Decision-Making:** Your documentation should demonstrate informed design trade-offs and clear articulation of your approach.
6. **Testing Strategy:** How comprehensive and practical is your proposed validation framework?

### **Communication**
7. **Final Presentation:** 45-minute code review and discussion where you present your solution, rationale, and lessons learned.

---

## Bonus Challenges (Optional)

Choose one or more to demonstrate advanced capabilities:

- **Comprehensive Evaluation Framework:** Implement automated testing with predefined engineering questions, domain expertise validation using MLflow evaluation, and performance metrics logging.

- **Human-in-the-Loop Integration:** Incorporate mechanisms for handling low-confidence scenarios with human oversight.

- **Enterprise Scalability:** Develop and document a detailed strategy for scaling the solution to handle thousands of documents with frequent updates.

---

## Getting Started

1. **Environment Setup:** Set up a Python virtual environment and install required dependencies
2. **Architecture Planning:** Design your RAG and agent approach before coding
3. **Iterative Development:** Build incrementally across the three tiers
4. **Documentation:** Maintain clear documentation throughout development
5. **Testing:** Validate your solution with the provided test queries

## Support & Questions

- **Clarifications:** Don't hesitate to ask about requirements or expectations
- **Technical Choices:** Document any technical decisions and trade-offs in your README

---

**Timeline:** You have **10 days** to complete this challenge. We recommend distributing your 8-10 hours across the timeline to allow for iterative development and refinement.

**Submission:** Provide repository access with complete documentation. The solution should be runnable locally with clear setup instructions.

Good luck! We're excited to see your approach to building production-ready AI engineering solutions.
