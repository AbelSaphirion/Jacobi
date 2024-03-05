# Jacobi
Library for calculating eigenvalues and eigenvectors
## Instalation
You can install this package with poetry:
```
poetry install
```
Or with pip:
```
pip install -r requirements.txt
pip install jacobi
```
## Usage
usage: jacobi.py [-h] [-e EPSILON] [-t THREADS] filepath
```
Jacobi calc.

positional arguments:
  filepath              Path to a csv file.

options:
  -h, --help            show this help message and exit
  -e EPSILON, --epsilon EPSILON
                        Calculation error.
  -t THREADS, --threads THREADS
                        Number of parallel threads.
```
