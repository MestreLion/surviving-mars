# This file is part of project <https://github.com/MestreLion/surviving-mars>
# Copyright (C) 2024 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
# License: GPLv3 or later, at your choice. See <http://www.gnu.org/licenses/gpl>
# -----------------------------------------------------------------------------
# Inspired by https://venthur.de/2021-03-31-python-makefiles.html

# To customize defaults such as PYTHON
-include .env

## SLUG: Short project name, for PyPI upload and generated files prefix
SLUG     ?= survivingmars
## EXEC: Project main executable, most likely $(SLUG). May be used in run and profile
EXEC     ?= sensortowers-heatmap
## PACKAGE: Project main package, most likely $(SLUG). To check if installed
PACKAGE  ?= sensortowers_heatmap
## EXTRAS: Extra packages to install for development (via `pip install -e .[$EXTRAS]`)
EXTRAS   ?= qt
## PYTHON: System python, used to create the virtual environment
PYTHON   ?= python3
## ENV_DIR: Path to the virtual environment, absolute or relative to current dir
ENV_DIR  ?= venv
## PROF_DIR: Path to profiling dir, where .pstats and .dot files are generated
PROF_DIR ?= profile

# Derived vars:
# path to virtual environment bin dir
venv     := $(ENV_DIR)/bin
# path to virtual environment python executable
python   := $(venv)/python
# path to virtual environment pip executable
pip      := $(venv)/pip
# path to the var dir inside the venv, used to keep target marker files
var_dir  := $(ENV_DIR)/var
# path to the 'venv created' marker file
venv_created      := $(var_dir)/venv-created
# path to the 'package installed' marker file
package_installed := $(var_dir)/installed-$(PACKAGE)
# paths to profiling stats file and dot graph
now      := $(shell date '+%Y%m%d%H%M%S')
pstats   := $(PROF_DIR)/$(SLUG)_$(now).pstats
dotgraph := $(PROF_DIR)/$(SLUG)_$(now).dot

# Often-customized vars:
# how to invoke the project (main) executable, for 'make run' and 'make profile'
# common options: $(venv)/$(EXEC); $(python) -m $(PACKAGE); $(SLUG)
run_cmd  := $(python) -m $(PACKAGE)
# command when profiling, most likely $(run_cmd) --demo or --benchmark
prof_cmd := $(run_cmd)

# -----------------------------------------------------------------------------
.DEFAULT_GOAL := help

## - run: run project or its main executable
run: install
	exec $(run_cmd)

## - install: install the project in editable mode in a virtual environment
install: $(package_installed)

## - build: build sdist and wheel packages using PyPA's `build` module
build: venv
	$(python) -m build

## - upload: upload built packages to PyPI using `twine`
upload: venv build
	$(venv)/twine upload --repository $(SLUG) -- dist/*

## - profile: Generate and open Dot graph at $PROF_DIR with `cProfile` and `gprof2dot`
profile: $(dotgraph)
$(PROF_DIR)/.gitignore:
	mkdir -p -- $(PROF_DIR)
	echo '*' > $(PROF_DIR)/.gitignore
$(dotgraph): $(venv)/gprof2dot $(PROF_DIR)/.gitignore
	$(python) -m cProfile -o $(pstats) $(prof_cmd)
	$(venv)/gprof2dot -f pstats -o $@ -- $(pstats)
	xdg-open $@

## - system-packages: apt-install system pre-dependencies `python3-{tk,dev,venv}`
system-packages:
	# I'm *sure* there's a better way of doing this... my make-fu is weak, PRs welcome!
	$(call is_installed,python3-tk)               || sudo apt install python3-tk
	$(call is_installed,$(notdir $(PYTHON))-dev)  || sudo apt install $(notdir $(PYTHON))-dev
	$(call is_installed,$(notdir $(PYTHON))-venv) || sudo apt install $(notdir $(PYTHON))-venv
define is_installed
	[ -n "$(shell dpkg-query --showformat='$${Version}' --show "${1}" 2>/dev/null || true)" ]
endef

.PHONY: run install format check build upload profile system-packages
# -----------------------------------------------------------------------------

$(venv_created):
	$(PYTHON) -m venv $(ENV_DIR)
	$(python) -m pip --disable-pip-version-check install --upgrade pip
	mkdir -p -- $$(dirname $@)
	touch -- $@

$(package_installed): | venv
	$(pip) install --upgrade -e .[$(EXTRAS)]
	touch -- $@

## - venv: create a virtual environment in $ENV_DIR, by default `./venv`
venv: $(venv_created)

## - venv-clean: remove the virtual environment
venv-clean:
	rm -rf $(ENV_DIR)

## - python: run Python interactive interpreter
python: venv
	exec $(python)

## - ipython: run IPython interactive interpreter
ipython: $(venv)/ipython
	exec $(venv)/ipython

## - bash: run bash subshell in the virtual environment
bash: venv
	. $(venv)/activate && exec bash

## - clean: remove build artifacts
clean:
	rm -rf *.egg-info

## - clean-all: remove build artifacts and the virtual environment
clean-all: clean venv-clean

## - help: display this message
help:
	@echo "Available env vars and targets:"
	@sed -n 's/^.*##[ ]//p' Makefile

$(venv)/%: | venv
	$(pip) install --upgrade $*
	touch -- $@

.PHONY: install-dev venv venv-clean python ipython bash clean clean-all help
