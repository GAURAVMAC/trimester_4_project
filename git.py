import subprocess, socket

m = input('Commit Message:')

r = 'origin'
b = 'master'

if socket.gethostbyname(socket.gethostname()) == '127.0.0.1':
    print('No Internet Connection')
else:
    subprocess.call(f'git add .')
    subprocess.call(f'git commit -m "{m}"')
    subprocess.call(f'git push -u {r} {b}')