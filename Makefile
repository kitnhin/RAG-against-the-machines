#index
MAX_CHUNK_SIZE = 2000
OVERLAP = 200

#search
QUERY = What versioning scheme does vLLM use?
k = 10

#search dataset
SEARCH_DATASET_PATH = datasets/datasets_public/public/AnsweredQuestions/dataset_code_public.json
SEARCH_DATASET_OUTPUT = data/output/search_results

#momo search
SEARCH_INPUT_PATH = datasets/datasets_public/public/AnsweredQuestions/dataset_code_public.json
SEARCH_OUTPUT_PATH = data/output/search_results/dataset_code_public.json
THRESHOLD = 0.5

#answer

index:
	python -m student index --max_chunk_size $(MAX_CHUNK_SIZE) --overlap $(OVERLAP)

search:
	python -m student search --query "$(QUERY)" --k $(k)

search_dataset:
	python -m student search_dataset --dataset_path $(SEARCH_DATASET_PATH) --k $(k) --save_directory $(SEARCH_DATASET_OUTPUT)

momo_search:
	./moulinette/moulinette-ubuntu evaluate_student_search_results ${SEARCH_OUTPUT_PATH} ${SEARCH_INPUT_PATH} --k $(k) --max_context_length 2000 --threshold $(THRESHOLD)

answer:
	python -m student answer --query "$(QUERY)" --k $(k)