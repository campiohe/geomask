
venv:
	ASDF_PYTHON_VERSION=3.11.6 python -m venv .venv


exe:
	gcc src/masker.c -o bin/masker.o
