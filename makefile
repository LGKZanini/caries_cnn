PYTHON_INTERPRETER = python3

docker_local_gpu_0:
	sudo docker run --gpus 1 -v /home/luiz/Documents/caries_cnn/data/aug:/home/luiz/app/data -ti caries-cnn:train python3 main.py 8 10 0

