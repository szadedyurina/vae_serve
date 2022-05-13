#!/bin/bash

celery -A server:celery_app worker --loglevel=WARNING -P solo --concurrency=2