import os

env_content = """IBM_API_KEY=Frmhy35WKITsOI5MNKJvJjNza8ntWbjRYb4oDR984yaS
IBM_PROJECT_ID=9e75ca01-f542-44e2-9d38-09aee47ed76b
IBM_URL=https://eu-de.ml.cloud.ibm.com
FLASK_SECRET_KEY=nutrigenie-secret-2024
FLASK_DEBUG=false
PORT=5000
"""

with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content)

print("SUCCESS: .env file updated!")
print()

# Verify
with open('.env', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            key, _, val = line.partition('=')
            if key in ('IBM_API_KEY', 'IBM_PROJECT_ID', 'IBM_URL'):
                display = val[:15] + '...' if len(val) > 15 else val
                print(f"  {key} = {display}  [len={len(val)}]")
