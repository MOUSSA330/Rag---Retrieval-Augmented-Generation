.PHONY: install run debug clean lint lint-strict index search search_dataset answer answer_dataset evaluate server

install:
	uv sync

run:
	uv run python3 -m src

debug:
	uv run python3 -m pdb -m src 

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 . || true
	uv run mypy src/ --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Automation Rules CLI

index:
	uv run python3 -m src index --max_chunk_size 2000

search:
	uv run python3 -m src search "What activation formats does the fused batched MoE layer return in vLLM?" --k 10

search_dataset:
	uv run python3 -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
	uv run python3 -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

answer:
	uv run python3 -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" --k 10

answer_dataset:
	uv run python3 -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions

evaluate:
	uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
	uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions//dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

index-incremental:
	uv run python -m src index_incremental --max_chunk_size 2000

server:
	uv run -m src serve