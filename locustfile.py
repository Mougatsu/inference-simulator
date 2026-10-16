"""Stress test: each Locust user is one client firing random dataset prompts back-to-back.

UI:       locust --host http://localhost:8000          (open http://localhost:8089)
Headless: locust --host http://localhost:8000 --headless -u 64 -r 16 -t 60s
"""
import random

from locust import HttpUser, task


class Client(HttpUser):
    @task
    def generate(self):
        self.client.post("/generate", json={"prompt_id": random.randrange(1000)})
