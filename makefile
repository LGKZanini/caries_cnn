PYTHON_INTERPRETER = python3

docker_local_build:
	sudo docker build -f Dockerfile -t caries-cnn:train .

docker_local_gpu_0:
	sudo docker run --gpus 1 -v /home/luiz/Documents/caries_cnn/data/aug:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 8 35 0

docker_server_gpu_0:
	sudo docker run --gpus 1 -v /home/luizzanini/data/aug/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py 16 40 0
