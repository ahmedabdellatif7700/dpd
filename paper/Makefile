# The paper's PDF, built in the TeX Live container: nothing installed on the machine (README.md).

IMAGE ?= texlive/texlive:latest-full
TEX   := docker run --rm -u $(shell id -u):$(shell id -g) -e HOME=/tmp -v $(CURDIR):/paper -w /paper $(IMAGE)

.PHONY: pdf clean
pdf:
	$(TEX) latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(TEX) latexmk -C main.tex
