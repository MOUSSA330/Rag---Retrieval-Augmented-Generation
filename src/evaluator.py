from src.models import MinimalSource, RagDataset, StudentSearchResults


def compute_iou(
    source_a: MinimalSource, source_b: MinimalSource
) -> float:  # -> Intersection over Union
    """
    Compute the Intersection over Union of two character ranges.

    Args:
        source_a: First source.
        source_b: Second source.

    Returns:
        IoU between 0 and 1, or 0 if the sources are in different files.
    """
    if source_a.file_path != source_b.file_path:
        return 0.0

    start_a, end_a = (
        source_a.first_character_index,
        source_a.last_character_index,
    )
    start_b, end_b = (
        source_b.first_character_index,
        source_b.last_character_index,
    )

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(0, intersection_end - intersection_start)

    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union <= 0:
        return 0.0

    return intersection / union


IOU_THRESHOLD = 0.05


def is_match(retrieved: MinimalSource, ground_truth: MinimalSource) -> bool:
    """
    Tell whether a retrieved source overlaps a ground-truth source enough.

    Args:
        retrieved: Source returned by the retriever.
        ground_truth: Expected source.

    Returns:
        True if the IoU reaches IOU_THRESHOLD.
    """
    return compute_iou(retrieved, ground_truth) >= IOU_THRESHOLD


def recall_at_k_for_question(
    retrieved_sources: list[MinimalSource],
    ground_truth_sources: list[MinimalSource],
    k: int,
) -> float:
    """Compute recall@k for one question.

    Args:
        retrieved_sources: Ranked sources returned by the retriever.
        ground_truth_sources: Expected sources.
        k: Only the first k retrieved sources are considered.

    Returns:
        Share of expected sources found in the top-k, or 1.0 if none
        are expected.
    """
    if not ground_truth_sources:
        return 1.0

    top_k_sources = retrieved_sources[:k]

    found_count = 0
    for gt_source in ground_truth_sources:

        match_lqinah = False

        for retrieved in top_k_sources:
            if is_match(retrieved, gt_source):
                match_lqinah = True
                break

        if match_lqinah:
            found_count += 1

    return found_count / len(ground_truth_sources)


def evaluate(
    student_results: StudentSearchResults,
    ground_truth: RagDataset,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[int, float]:
    """Average recall@k over every question of a search results file.

    Args:
        student_results: Output of search_dataset.
        ground_truth: Dataset containing the expected sources.
        k_values: Values of k to evaluate.

    Returns:
        Mapping from each k to its average recall.

    Raises:
        ValueError: If the results are empty or were generated with a k smaller
            than the largest value in k_values.
    """

    if not student_results.search_results:
        raise ValueError("student_results is empty, nothing to evaluate")

    max_k = max(k_values)
    if student_results.k < max_k:
        raise ValueError(
            f"Results were generated with k={student_results.k}, but "
            f"evaluation needs k>={max_k}. "
            f"Run search_dataset again with --k {max_k} or more."
        )

    ground_truth_map = {
        q.question_id: q.sources
        for q in ground_truth.rag_questions
        if hasattr(q, "sources")
    }

    recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

    for result in student_results.search_results:
        gt_sources = ground_truth_map.get(result.question_id)
        if gt_sources is None:
            print(
                f"Warning: question_id {result.question_id} not found "
                "in ground truth, skipping !! :)"
            )
            continue

        for k in k_values:
            score = recall_at_k_for_question(
                retrieved_sources=result.retrieved_sources,
                ground_truth_sources=gt_sources,
                k=k,
            )
            recall_scores[k].append(score)

    final_averages = {}

    for k, scores in recall_scores.items():
        if len(scores) > 0:
            average = sum(scores) / len(scores)
            final_averages[k] = average
        else:
            final_averages[k] = 0.0

    return final_averages
