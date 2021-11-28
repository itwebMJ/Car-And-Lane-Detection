from flask import Blueprint, request, redirect, render_template, session
import models.board as b

bp = Blueprint("board", __name__, url_prefix = "/board")
board_service = b.cBoard_service()

@bp.route("/")
def board() :
    bd = board_service.Get_board("writer", session['id'])
    page = request.args.get("page", 1, int)

    return render_template("/board/board.html", bd = bd, page = page)

@bp.route("/insert")
def insert_form() :
    return render_template("/board/createBoard.html")

@bp.route("/insert", methods = ["POST"])
def insert() :
    title = request.form["title"]
    content = request.form["content"]

    bd = b.cBoard(writer = session['id'], title = title, content = content)
    board_service.Add_board(bd)
    return redirect("/board/")
