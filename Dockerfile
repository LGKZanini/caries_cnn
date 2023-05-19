FROM nvidia/cuda:12.1.1-base-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m luiz

RUN chown -R luiz:luiz /home/luiz/

COPY --chown=luiz . /home/luiz/app/

USER luiz

RUN cd /home/luiz/app/ && pip3 install -r requirements.txt

WORKDIR /home/luiz/app