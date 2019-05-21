install:
	pip install numpy==1.16.2
	pip install -r requirements.txt

train_gan:
	python -m train_gan --real_dataset ./data/unn_data.pickle --real_labels ./data/unn_labels.csv