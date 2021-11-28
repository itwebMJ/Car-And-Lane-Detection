import models.member as m
from flask import request


member_service = m.cMember_service()

def Insert_db() :
    member_id = request.form["id"]
    member_pw = request.form["pwd"]
    name = request.form["name"]
    mobile = request.form["mobile"]
    email = request.form["email"]
    member_service.Add_member(m.cMember(member_id, member_pw, name, mobile, email))
    return 

def Check_login() :
    member_id = request.form["id"]
    member_pw = request.form["pwd"]
    return member_service.Check_id(member_id, member_pw)
      
def Update_pwd(member_id) :
    new_pwd = request.form["pwd"]
    member_service.Edit_member(new_pwd, member_id)
    return

def Update_my_info(member_id) :
    new_name = request.form["name"]
    new_mobile = request.form["mobile"]
    member_service.Edit_member_info(new_name, new_mobile, member_id)
    return