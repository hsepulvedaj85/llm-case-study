from src.embedder import Embedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Generate evaluation report
print(f"Generate evaluation report")

def eval_file(questions, expected_answers, predicted_answers):
    """
    Evaluates the performance of a RAG (Retrieval-Augmented Generation) system by comparing 
    predicted answers against expected answers using cosine similarity.

    This function takes lists of questions, their expected (true) answers, and the answers 
    generated by the RAG system. It calculates the semantic similarity between the predicted and 
    expected answers by embedding them and computing cosine similarity scores. Finally, it 
    generates a detailed evaluation report including individual question results and an overall 
    accuracy score, saving it to a file.

    Args:
        questions (list[str]): A list of the input questions.
        expected_answers (list[str]): A list of the human-verified or true answers
                                      corresponding to the questions.
        predicted_answers (list[str]): A list of the answers generated by the RAG system
                                       for each question.

    Returns:
        None: This function does not return a value but prints progress and
              saves an evaluation report to `./data/evaluation_report.txt`.

    Example:
        >>> # Assuming you have lists of questions, expected answers, and predicted answers
        >>> # from your RAG system's run.
        >>> questions = ["What is the capital of France?", "Who painted the Mona Lisa?"]
        >>> expected_answers = ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."]
        >>> predicted_answers = ["The capital of France is Paris.", "The Mona Lisa was painted by Leonardo da Vinci."]
        >>> eval_file(questions, expected_answers, predicted_answers)
        # This will embed the answers, compute similarities, and generate a report.
    """
    print(f"\nEmbed predicted and expected answers:")
    # Embed predicted and expected answers
    embedder = Embedder()
    true_embeddings = [embedder.embed_texts(e) for e in expected_answers] 
    pred_embeddings = [embedder.embed_texts(p) for p in predicted_answers]

    # Compute cosine similarity scores
    print("\nCompute cosine similarity scores...")
    similarities = cosine_similarity(pred_embeddings, true_embeddings)
    similarity_scores = [similarities[i, i] for i in range(len(similarities))]
    accuracy = np.mean(similarity_scores)
    print(f"Accuracy: {accuracy}")
    # Embed predicted and expected answers
    report_lines = []
    for i, question in enumerate(questions):
        report_lines.append(f"Q{i+1}: {question}")
        report_lines.append(f"Expected: {expected_answers[i]}")
        report_lines.append(f"Predicted: {predicted_answers[i]}")
        report_lines.append(f"Cosine Similarity: {similarity_scores[i]:.4f}\n")

    report_lines.append(f"\nOverall Accuracy (Avg Cosine Similarity): {accuracy:.4f}")

    # Save report to file
    print(f"Save report to file")
    report_path = "./data/evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Finaliza Eval") 