import requests

BASE = 'http://127.0.0.1:5000'
ROUTES = ['/', '/disease', '/detect', '/quality', '/camera', '/vqa']

for r in ROUTES:
    url = BASE + r
    try:
        resp = requests.get(url, timeout=10)
        print('---', r, resp.status_code)
        txt = resp.text.replace('\n','\\n')
        print(txt[:800])
    except Exception as e:
        print('---', r, 'ERROR', e)
