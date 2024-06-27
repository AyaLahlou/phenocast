install:
	pip install -e .[lint,test,dev]

#test:
	pytest tests/

# lint, format, and check all files
lint:
	pre-commit run --all-files

