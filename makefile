PYTHON_INTERPRETER = python3

docker_build:
	sudo docker build -f Dockerfile -t caries-cnn:train .

experiment_1:
	sudo docker run --gpus all -v /home/luizzanini/data/:/home/luiz/app/data --env-file .env1 -ti caries-cnn:train python3 main.py

experiment_2:
	sudo docker run --gpus all -v /home/luizzanini/data/:/home/luiz/app/data --env-file .env2 -ti caries-cnn:train python3 main.py
