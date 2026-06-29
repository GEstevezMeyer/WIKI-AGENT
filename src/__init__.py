import os
from multiprocessing import cpu_count
from dotenv import load_dotenv

load_dotenv()

os.environ["cpu_count"] = str(cpu_count())
token = os.getenv("HF_TOKEN")
