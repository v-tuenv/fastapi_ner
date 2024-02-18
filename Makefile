IMAGE_NAME = "fastapi-ner-sever"
VERSION = v1.0

HOST = localhost
PORT = 8001
MODEL_PATH=model_hub/deberta_fintune_ner_17_05

build:
	./build_docker.sh cpu ${IMAGE_NAME} ${MODEL_PATH}

run:
	[ $(docker ps -a | grep ${IMAGE_NAME}-${VERSION}) ] | docker rm -f ${IMAGE_NAME}-${VERSION}
	docker run --name ${IMAGE_NAME}-${VERSION} -it -p8001:80  ${IMAGE_NAME}

test:
	python3 clients/test_model.py
