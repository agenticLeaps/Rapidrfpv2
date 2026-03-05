#!/usr/bin/env python3
"""
Celery Configuration for NodeRAG Queue Processing
"""

from celery import Celery
import os
from src.config.settings import Config

# Redis connection URL
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'noderag_queue',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['src.queue.tasks']
)

# Celery configuration - Optimized for solo pool on 2GB RAM
celery_app.conf.update(
    # Task routing
    task_routes={
        'src.queue.tasks.process_document_task': {'queue': 'document_processing'},
        'src.queue.tasks.cleanup_task': {'queue': 'cleanup'},
    },

    # Worker settings for solo pool (no child processes)
    # NOTE: worker_max_tasks_per_child does NOT work with solo pool
    worker_prefetch_multiplier=1,  # Only fetch one task at a time
    task_acks_late=True,  # Acknowledge task only after completion

    # Task settings - Use JSON to avoid serialization issues
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task time limits (prevent hanging)
    task_time_limit=1800,  # 30 minutes hard limit
    task_soft_time_limit=1500,  # 25 minutes soft limit

    # Queue settings
    task_default_queue='document_processing',
    task_default_exchange='noderag',
    task_default_exchange_type='direct',
    task_default_routing_key='document_processing',

    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        'visibility_timeout': 3600,
    },

    # Broker connection settings - prevent disconnections
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    broker_heartbeat=30,  # Send heartbeat every 30 seconds

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task annotations for memory-sensitive operations
celery_app.conf.task_annotations = {
    'src.queue.tasks.process_document_task': {
        'rate_limit': '2/m',  # 2 tasks per minute max - prevent overload
        'time_limit': 1800,
        'soft_time_limit': 1500,
    }
}

if __name__ == '__main__':
    celery_app.start()