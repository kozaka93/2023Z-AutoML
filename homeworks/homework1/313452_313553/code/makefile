setup:
	pip install --extra-index-url https://pypi.fury.io/arrow-nightlies/ \
	--prefer-binary --pre pyarrow
	pip install -r requirements.txt
	python hpo_tune/init.py 

run_experiments:
	python hpo_tune/bin.py