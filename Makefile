SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-print-directory

CURRENT_DIR := $(shell pwd)

.DEFAULT_GOAL := help

.PHONY: help bash local download-weights


help:
	@echo "QuadTreeAttention"
	@grep -E '^[a-zA-Z_0-9%-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-30s %s\n", $$1, $$2}'

local: ## Build docker image locally
	docker build -f $(shell pwd)/Dockerfile -t qta .

bash: ## Start a bash session inside the container
	docker run --gpus all \
	-v $(shell pwd)/data/:/workspace/data \
	-v $(shell pwd)/match_pair.py:/workspace/match_pair.py \
	--rm -it --entrypoint bash qta

download-weights: ## Download weights for feature matching
	mkdir -p weights
	cd weights
	wget https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/indoor.ckpt
	wget https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt

install: ## Install packages and dependencies
	pip install -r requirements.txt && \
	pip install -e ./QuadTreeAttention/ && \
	pip install -e ./FeatureMatching/