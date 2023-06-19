PYTHON_INTERPRETER = python3

docker_build:
	sudo docker build -f Dockerfile -t caries-cnn:train .

experiment_classify_0:
	sudo docker run --gpus 1 -v /home/luizzanini/data/aug_v3/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 cnn simple

experiment_classify_1:
	sudo docker run --gpus all -v /home/luizzanini/data/aug_v3/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 cnn simple

experiment_classify_rotate_0:
	sudo docker run --gpus all -v /home/luizzanini/data/aug_v3/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 cnn rotate

experiment_classify_rotate_1:
	sudo docker run --gpus all -v /home/luizzanini/data/aug_v3/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 cnn rotate

experiment_classify_jigsaw_0:
	sudo docker run --gpus all -v /home/luizzanini/data/aug_v3/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 cnn jigsaw

experiment_classify_jigsaw_1:
	sudo docker run --gpus all -v /home/luizzanini/data/aug_v3/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 cnn jigsaw

experiment_rotate_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 rotate simple

experiment_rotate_1:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 rotate simple

experiment_jigsaw_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 jigsaw simple

experiment_jigsaw_1:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 jigsaw simple
