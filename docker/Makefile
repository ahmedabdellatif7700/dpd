# The dpd stage's container: common tasks. Run `make -C docker` (or `make help`) for the list.
# After the user's Jetson CUDA template (jetson-llama-cuda/Makefile): the same Makefile works
#   on the Jetson   -> build and run the image
#   on the laptop   -> run the same targets on the Jetson over SSH (JETSON=orinnano8s)

JETSON     ?= orinnano8s
JETSON_DIR ?= dpd

COMPOSE := docker compose -f compose.yaml

.DEFAULT_GOAL := help
.PHONY: help build check shell ps logs down clean jetson-build jetson-check

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-z-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ On the Jetson

build: ## Build the image arof-dpd:local (compiles cpp/ for sm_87)
	@mkdir -p ../results/models ../logs
	$(COMPOSE) build

check: build ## Run the image's GPU self-test (gpu_check) with the NVIDIA runtime
	$(COMPOSE) run --rm dpd_bb gpu_check

shell: ## A shell in the image, GPU attached
	$(COMPOSE) run --rm dpd_bb bash

ps: ## Show the container's status
	$(COMPOSE) ps

logs: ## Follow the container's output
	$(COMPOSE) logs -f

down: ## Stop and remove the container
	$(COMPOSE) down

clean: ## Stop and remove the built image (keeps results/ and logs/)
	$(COMPOSE) down --rmi local

##@ On the laptop: the same on the Jetson (JETSON=orinnano8s, JETSON_DIR=dpd)

jetson-build: ## Build the image on the Jetson over SSH
	ssh $(JETSON) 'cd $(JETSON_DIR) && make -C docker build'

jetson-check: ## Run the GPU self-test on the Jetson over SSH
	ssh $(JETSON) 'cd $(JETSON_DIR) && make -C docker check'
