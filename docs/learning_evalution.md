1. Evaluation can be divided into 2 kind of Evaluation
    - Model Evaluation
    - LLM Application Evaluation

2. Whenever a new LLM model is release its generally tested on the following capabilities
    - Reasoning
    - General Knowledge
    - Maths
    - Coding
    - Instruction Following
    - Long Context
    - Multi model Understanding
    - Tool Use

3. Following datasets/benchmark are there for the specific capability
    - General Knowledge/Reasoning  - MMLU
    - Maths - GSM8k
    - Coding - HumanEval,SWE-Bench
    - Instruction Following - IFEval
    - Long Context - Needle-in-a-Haystack
    - Multi modal Capabilities - MMMu

4. Application Evaluation Metrics
    4.1 Task Specific Metrics
        - Application Quality
            - Correctness/accuracy
            - Relevance
            - Completeness
            - Instruction Following

        - RAG Specific Quality
            - Context Relevance
            - Retrieval Recall
            - Groudedness/Faithfulness 
            - Citation Accuracy
        
        - Agent/Tool-use Quality
            - Tool Selection
            - Parameter Correctness
            - Task Completeness
            - Error Recovery

        - Multi Turn Conversation Quality
            - Context Retention/Memory
            - Classification Behavior

    4.2. Safety
        - Toxicity
        - Harmful Content
        - Bias/fairness
        - PII leakage/Masking
        - Prompt Injection/Jailbreak resistance

    4.3. Operational
        - End-to-End Latency
        - Cost Per Request
        - Token efficiency
        - Error / Failure Rate
        - Latency Under Load

# Doubts
What is this concept? like ways to evaluation ?
    1. What is this Programmatic / Deterministic
    2. Human
    3. Model graded(LLM as judge)