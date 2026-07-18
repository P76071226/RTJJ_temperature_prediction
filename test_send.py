import sys
sys.path.insert(0, '/data/data/com.termux/files/home/hermes-agent')
from hermes_tools import send_message
print("Calling send_message...")
result = send_message(target='telegram', message='test', media='')
print("Result:", result)