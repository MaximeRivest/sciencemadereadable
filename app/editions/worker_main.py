"""Separate resource-limited worker so a failed PDF cannot take down the HTTP API."""

import os
import signal
from service import Worker
from store import Store

store = Store(os.environ.get("SMR_READING_STATE", "/var/lib/smr-editions"))
worker = Worker(store)


def stop(signum, frame):
    worker.stop.set()
    worker.terminate()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker.loop()
