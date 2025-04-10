import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("HF_API_KEY")
print("Key exists:", bool(key))  # Should print "True"