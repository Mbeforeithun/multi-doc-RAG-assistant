# import json
# from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv()

# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings

# from src.ingestion import get_vectorstore
# from src.retrieval import get_hybrid_retriever, get_router
# from src.chain import get_rag_chain, get_query_rewriter, get_faithfulness_checker, answer_question


# def load_eval_set(path="eval/eval_set.json"):
#     return json.loads(Path(path).read_text())


# def build_ragas_dataset():
#     vectorstore = get_vectorstore()
#     hybrid_retriever = get_hybrid_retriever(vectorstore)
#     router = get_router()
#     rag_chain = get_rag_chain()
#     rewriter = get_query_rewriter()
#     checker = get_faithfulness_checker()

#     eval_set = load_eval_set()

#     questions, answers, contexts, ground_truths = [], [], [], []

#     for item in eval_set:
#         if not item["answerable"]:
#             continue  # RAGAS's metrics assume there IS a correct answer to check against

#         print(f"Running: {item['question']}", flush=True)
#         result = answer_question(
#             item["question"], [], rag_chain, rewriter, checker,
#             vectorstore, router, hybrid_retriever,
#         )

#         questions.append(item["question"])
#         answers.append(result["answer"])
#         contexts.append([doc.page_content for doc in result["sources"]])
#         # Using the expected key phrases joined as a rough "ground truth" stand-in
#         ground_truths.append(", ".join(item.get("expected_answer_contains", [])))

#     return Dataset.from_dict({
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts,
#         "ground_truth": ground_truths,
#     })


# if __name__ == "__main__":
#     dataset = build_ragas_dataset()

#     judge_llm = ChatGroq(model="openai/gpt-oss-120b")
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

#     result = evaluate(
#         dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
#         llm=judge_llm,
#         embeddings=embeddings,
#     )

#     print(result)
#     result.to_pandas().to_csv("eval/ragas_results.csv", index=False)
#     print("\nSaved detailed results to eval/ragas_results.csv")

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.ingestion import get_vectorstore
from src.retrieval import get_hybrid_retriever, get_router
from src.chain import get_rag_chain, get_query_rewriter, get_faithfulness_checker, answer_question

REFUSAL_PHRASE = "i don't know"


def load_eval_set(path="eval/eval_set.json"):
    return json.loads(Path(path).read_text())


def check_hit(sources, expected_section):
    if expected_section is None:
        return None
    return any(expected_section.split(" ", 1)[-1].lower() in doc.page_content.lower() for doc in sources)


def check_answer_correct(answer, expected_phrases):
    if not expected_phrases:
        return None
    answer_lower = answer.lower()
    return any(phrase.lower() in answer_lower for phrase in expected_phrases)


def run_eval():
    vectorstore = get_vectorstore()
    hybrid_retriever = get_hybrid_retriever(vectorstore)
    router = get_router()
    rag_chain = get_rag_chain()
    rewriter = get_query_rewriter()
    checker = get_faithfulness_checker()

    eval_set = load_eval_set()
    results = []

    for item in eval_set:
        print(f"Running: {item['question']}", flush=True)
        try:
            result = answer_question(
                item["question"], [], rag_chain, rewriter, checker,
                vectorstore, router, hybrid_retriever,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue

        hit = check_hit(result["sources"], item.get("expected_section"))
        correct = check_answer_correct(result["answer"], item.get("expected_answer_contains", []))
        refused = REFUSAL_PHRASE in result["answer"].lower()

        results.append({
            "question": item["question"],
            "answerable": item["answerable"],
            "hit": hit,
            "correct": correct,
            "is_faithful": result["is_faithful"],
            "faithfulness_reason": result["faithfulness_reason"],
            "answer": result["answer"],
            "refused": refused,
            "strategy": result["strategy"],
        })

    return results


def summarize(results):
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    hit_rate = sum(1 for r in answerable if r["hit"]) / len(answerable) if answerable else 0
    correctness = sum(1 for r in answerable if r["correct"]) / len(answerable) if answerable else 0
    faithfulness = sum(1 for r in answerable if r["is_faithful"]) / len(answerable) if answerable else 0
    refusal_accuracy = sum(1 for r in unanswerable if r["refused"]) / len(unanswerable) if unanswerable else 0

    print("\n===== EVAL SUMMARY =====")
    print(f"Answerable questions: {len(answerable)}")
    print(f"  Hit Rate@3:        {hit_rate:.0%}")
    print(f"  Answer correctness: {correctness:.0%}")
    print(f"  Faithfulness rate:  {faithfulness:.0%}")
    print(f"Unanswerable questions: {len(unanswerable)}")
    print(f"  Refusal accuracy:  {refusal_accuracy:.0%}")

    print("\n--- Failures ---")
    for r in answerable:
        if not r["hit"] or not r["correct"]:
            print(f"MISS: {r['question']} (hit={r['hit']}, correct={r['correct']}, strategy={r['strategy']})")
        if not r["is_faithful"]:
            print(f"UNFAITHFUL: {r['question']}")
            print(f"  Answer: {r['answer']}")
            print(f"  Reason: {r['faithfulness_reason']}")
    for r in unanswerable:
        if not r["refused"]:
            print(f"HALLUCINATED: {r['question']} (strategy={r['strategy']})")


if __name__ == "__main__":
    results = run_eval()
    summarize(results)