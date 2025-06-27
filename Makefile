build:
	docker build . -t ruperts

debug:
	docker run  --rm -it --entrypoint=/bin/bash ruperts:latest
