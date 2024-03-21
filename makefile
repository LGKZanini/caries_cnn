PYTHON_INTERPRETER = python3

docker_build:
	sudo docker build -f Dockerfile -t caries-cnn:train .

experiment:
	sudo docker run --gpus all -v /home/luizzanini/data/cbct_DL/:/home/luiz/app/data --env-file .env -ti caries-cnn:train python3 main.py
