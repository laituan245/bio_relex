#!/bin/bash
pip3 install -r $1/requirements.txt && python3 $1/biorelex_code.py $1/model.pt $2 $3 $4
