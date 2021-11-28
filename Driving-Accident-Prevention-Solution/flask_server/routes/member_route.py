from flask import Blueprint, render_template, request, redirect, session
import models.member as m
from routes.member_func import *



bp = Blueprint("member", __name__, url_prefix = "/member")
member_service = m.cMember_service()
pwd_flag = False

@bp.route('/sign_up')
def sign_up() :
    mem = member_service.Get_all()
    mem_lst = []
    for i in mem :
        lst = []
        lst.append(i.member_id)
        lst.append(i.email)
        mem_lst.append(lst)
    
    return render_template("/member/sign_up.html", mem = mem_lst)

@bp.route('/search_id')
def search_id() :
    mem = member_service.Get_all()
    mem_lst = []
    for i in mem :
        lst = []
        lst.append(i.name)
        lst.append(i.email)
        lst.append(i.member_id)
        mem_lst.append(lst)
    
    return render_template("/member/search_id.html", mem = mem_lst)

@bp.route('/search_pwd')
def search_pwd() :
    mem = member_service.Get_all()
    mem_lst = []
    for i in mem :
        lst = []
        lst.append(i.member_id)
        lst.append(i.email)  
        mem_lst.append(lst)
    
    return render_template("/member/search_pwd.html", mem = mem_lst)

@bp.route('/change_pwd')
def change_pwd() :
    mem_id = request.args.get("id", "0", str)
    session["tmp_id"] = mem_id
    global pwd_flag
    pwd_flag = True
    return redirect("/member/move_change_pwd")

@bp.route('/change_pwd', methods=["POST"])
def run_change_pwd() :
    Update_pwd(session["tmp_id"])
    return redirect("/")

@bp.route('/move_change_pwd')
def move_change_pwd() :
    global pwd_flag
    if pwd_flag :
        pwd_flag = False
        return render_template("/member/change_pwd.html")
    return redirect("/")


@bp.route("/insert", methods=["POST"]) 
def insert() :
    Insert_db()
    #print(member_id, member_pw, name, mobile, email)
    return redirect("/")


@bp.route("/login", methods = ["POST"])
def login() :
    session["id"] = request.form["id"]
    return render_template("/play.html")

@bp.route("/login")
def index() :
    return render_template("/play.html")


@bp.route("/my_info")
def my_info() :
    mem = member_service.Get_member_by_id(session["id"])
    mem_lst = [mem.member_id, mem.name, mem.mobile, mem.email]
    return render_template("/member/my_info.html", mem = mem_lst)

@bp.route("/update_my_info", methods = ["POST"])
def update_my_info() :
    Update_my_info(session["id"])
    return redirect("/member/my_info")


