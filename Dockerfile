FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m luiz

RUN chown -R luiz:luiz /home/luiz/

USER luiz

COPY ./requirements.txt /home/luiz/app/requirements.txt

RUN cd /home/luiz/app/ && pip3 install -r requirements.txt

COPY --chown=luiz . /home/luiz/app/

RUN chown -R luiz:luiz ./home/luiz/app/

WORKDIR /home/luiz/app