PYTHON_INTERPRETER = python3

docker_build:
	sudo docker build -f Dockerfile -t caries-cnn:train .

experiment_classify_0:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 64 200 0 cnn simple resnet18

experiment_classify_1:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 64 200 1 cnn simple resnet50

experiment_classify_3:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 64 200 0 cnn simple densenet121

experiment_classify_4:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 64 200 1 cnn simple VGG19

experiment_classify_rotate_0:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 cnn rotate

experiment_classify_rotate_1:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 cnn rotate

experiment_classify_jigsaw_0:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 cnn jigsaw

experiment_classify_jigsaw_1:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 cnn jigsaw

experiment_rotate_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 rotate simple

experiment_rotate_1:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 rotate simple

experiment_jigsaw_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 jigsaw simple

experiment_jigsaw_1:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 1 jigsaw simple

experiment_byol_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl_lighty/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 byol simple resnet50

experiment_simclr_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl_lighty/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 simclr simple resnet50

experiment_vicreg_0:
	sudo docker run --gpus all -v /home/luizzanini/data/ssl_lighty/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 32 100 0 vicreg simple resnet50
