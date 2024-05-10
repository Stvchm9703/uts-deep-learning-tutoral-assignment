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

run-obj-dtt:
	if test -f object-detection.nbconvert.ipynb; then \
		cp object-detection.nbconvert.ipynb backup/object-detection.nbconvert-$(shell date --iso=seconds).ipynb_bakcup;\
	fi;\
	 python -m jupyter nbconvert object-detection.ipynb --to notebook --execute; \

run-yolox-trainer:
	cd YOLOX;\
	python  tools/train.py -n yolox-nano -b 32 --fp16; \


run-detectron2-trainer:
	if test -f detectron2.nbconvert.ipynb; then \
		cp detectron2.nbconvert.ipynb backup/detectron2.nbconvert-$(shell date --iso=seconds).ipynb_bakcup;\
	fi;\
	 python -m jupyter nbconvert detectron2.ipynb --to notebook --execute; \


run-mmdetection:
	if test -f mmdetection-runner.nbconvert.ipynb; then \
		cp mmdetection-runner.ipynb backup/mmdetection-runner.nbconvert-$(shell date --iso=seconds).ipynb_bakcup;\
	fi;\
	 python -m jupyter nbconvert mmdetection-runner.ipynb --to notebook --execute; \