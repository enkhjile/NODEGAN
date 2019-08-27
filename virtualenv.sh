#!/bin/bash

sudo pip install virtualenvwrapper
echo export VIRTUALENVWRAPPER_PYTHON='/usr/bin/python' >> ~/.bashrc
source ~/.bashrc
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv venv & workon venv
pip install -r ~/NODEGAN/requirements.txt
echo source /usr/local/bin/virtualenvwrapper.sh >> ~/.bashrc
