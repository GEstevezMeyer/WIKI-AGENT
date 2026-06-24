import os
from multiprocessing import cpu_count

os.environ["cpu_count"] = str(cpu_count())