#!/bin/bash

sudo pip install virtualenvwrapper
echo export VIRTUALENVWRAPPER_PYTHON='/usr/bin/python' >> ~/.bashrc
echo source `which virtualenvwrapper.sh` >> ~/.bashrc
source `which virtualenvwrapper.sh`
# source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv venv
workon venv
pip install -r ~/NODEGAN/requirements.txt
