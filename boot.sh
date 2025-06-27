#!/bin/sh
python -m spacy download en_core_web_md
gunicorn app:app --bind 0.0.0.0:$PORT