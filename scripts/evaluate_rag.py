"""
Automated RAG Evaluation Script using Ragas and LangSmith.
Reads data/eval_dataset.json, runs the retrieval pipeline, and computes
Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
Results are saved to data/eval_results.json for the admin dashboard.
"""
import os
import json
import asyncio
import time
from pathlib import Path
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from app.config import get_settings
from app.rag.retriever import retrieve
from app.rag.prompt_builder import build_messages
from openai import AsyncOpenAI

s = get_settings()

# Set up environment for LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = s.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = s.langchain_project

# Provide API keys for Ragas underlying models
os.environ["OPENAI_API_KEY"] = s.openrouter_api_key
os.environ["OPENAI_API_BASE"] = s.openrouter_base_url

client = AsyncOpenAI(
    api_key=s.openrouter_api_key,
    base_url=s.openrouter_base_url
)


async def generate_answer(query: str, contexts: list[dict]) -> str:
    messages = build_messages(query, contexts, [])
    completion = await client.chat.completions.create(
        model=s.openrouter_model,
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )
    return completion.choices[0].message.content


async def run_evaluation():
    print("[Eval] Loading eval dataset...")
    dataset_path = Path("data/eval_dataset.json")
    if not dataset_path.exists():
        print("Dataset not found at data/eval_dataset.json")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    ground_truths = []
    answers = []
    contexts_list = []

    print(f"[Eval] Running pipeline for {len(data)} queries...")
    for item in data:
        q = item["question"]
        gt = item["ground_truth"]
        
        # Retrieve chunks
        chunks = await retrieve(q, k=s.top_k_chunks)
        contexts = [c["text"] for c in chunks]
        
        # Generate answer
        ans = await generate_answer(q, chunks)
        
        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts_list.append(contexts)
        
        # Rate limit protection for free tiers
        await asyncio.sleep(1.0)

    # Format for Ragas
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })

    print("[Eval] Executing Ragas metrics...")
    
    # Ragas uses Langchain models for evaluation
    eval_llm = ChatOpenAI(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        model="openai/gpt-4o-mini"
    )
    eval_embeddings = OpenAIEmbeddings(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        model="text-embedding-3-small"
    )

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )

    scores = {
        "faithfulness": result.get("faithfulness", 0.0),
        "answer_relevancy": result.get("answer_relevancy", 0.0),
        "context_precision": result.get("context_precision", 0.0),
        "context_recall": result.get("context_recall", 0.0),
        "timestamp": time.time()
    }
    
    print(f"[Eval] Results: {scores}")
    
    # Save results to file for the admin UI
    out_path = Path("data/eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
        
    print(f"[Eval] Saved results to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
