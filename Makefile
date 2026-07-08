#process file ("docs" or "code")
TYPE = docs

#index
MAX_CHUNK_SIZE = 2000
OVERLAP = 200

#search
QUERY = What versioning scheme does vLLM use?
k = 10

#search dataset
SEARCH_DATASET_PATH = datasets/datasets_public/public/AnsweredQuestions/dataset_$(TYPE)_public.json
SEARCH_DATASET_OUTPUT = data/output/search_results

#momo search
SEARCH_INPUT_PATH = datasets/datasets_public/public/AnsweredQuestions/dataset_$(TYPE)_public.json
SEARCH_OUTPUT_PATH = data/output/search_results/dataset_$(TYPE)_public.json
ifeq ($(TYPE), docs)
THRESHOLD = 0.8
else
THRESHOLD = 0.5
endif

#answer

index:
	uv run python -m student index --max_chunk_size $(MAX_CHUNK_SIZE) --overlap $(OVERLAP)

search:
	uv run python -m student search --query "$(QUERY)" --k $(k) --type $(TYPE)

search_dataset:
	uv run python -m student search_dataset --dataset_path $(SEARCH_DATASET_PATH) --k $(k) --save_directory $(SEARCH_DATASET_OUTPUT) --type $(TYPE)

momo_search:
	./moulinette/moulinette-ubuntu evaluate_student_search_results ${SEARCH_OUTPUT_PATH} ${SEARCH_INPUT_PATH} --k $(k) --max_context_length ${MAX_CHUNK_SIZE} --threshold $(THRESHOLD)

answer:
	uv run python -m student answer --query "$(QUERY)" --k $(k)

answer_dataset:
	uv run python -m student answer_dataset --dataset_path $(SEARCH_DATASET_PATH) --k $(k) --save_directory $(SEARCH_DATASET_OUTPUT)

index_and_search: clean_index index search_dataset momo_search

clean_index:
	rm -rf data/processed/

clean_output:
	rm -rf data/output/