install:
	poetry install

install-hooks:
	poetry run pre-commit install

repro:
	poetry run dvc repro

lint:
	poetry run pre-commit run --all-files

clean:
	rm -rf data/processed models metrics reports
