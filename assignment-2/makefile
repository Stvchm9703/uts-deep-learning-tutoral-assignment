jupyter = /home/ghost/anaconda3/envs/uts-dl/bin/jupyter
session:
	screen -DRRS uts-dl-sess;

run-test:
	conda activate uts-dl; \
	cp object-detection.ipynb object-detection.ipynb_bakcup; \
	jupyter nbconvert object-detection.ipynb --to notebook --execute; \

run-img-clf:
	if test -f image-classification.nbconvert.ipynb; then \
		cp image-classification.nbconvert.ipynb backup/image-classification.nbconvert-$(shell date --iso=seconds).ipynb_bakcup;\
	fi;\
	 python -m jupyter nbconvert image-classification.ipynb --to notebook --execute; \

