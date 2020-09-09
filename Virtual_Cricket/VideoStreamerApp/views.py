from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from .camera import MaskDetect


# Create your views here.


def index(request):
    runs = 1
    USER = True
    show_variables = dict(
        bot_runs=17,
        bot_runs_array=[],
        user_runs=18,
        user_runs_array=[],
        message=f'{"USER" if USER else "BOT"} WON BY {abs(runs)} RUNS'
    )
    return render(request, 'VideoStreamerApp/home.html', context=show_variables)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def mask_feed(request):
    return StreamingHttpResponse(gen(MaskDetect()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
