from flask import Flask, render_template
from simple_websocket_server import WebSocketServer, WebSocket
import socket, threading, pickle, struct
from server_func import *
import routes.member_route as mr
import routes.board_route as br
import models.member as m

app = Flask(__name__, template_folder = "templates")
app.config['SECRET_KEY'] = 'secret!'
app.register_blueprint(mr.bp)
app.register_blueprint(br.bp)
my_webServer_run = False
member_service = m.cMember_service()

@app.route('/')
def home() :
    global my_webServer_run
    if not my_webServer_run :
        my_webServer_run = True
        init_my_webServer()
        
    mem = member_service.Get_all()
    meminfo_lst = []
    for i in mem:
        lst = []
        lst.append(i.member_id)
        lst.append(i.member_pw)
        meminfo_lst.append(lst)
    return render_template("/index.html", mem = meminfo_lst)



if __name__ == '__main__':                              
    app.run(host = "172.31.38.121", port = 5050, debug = True)
    
    