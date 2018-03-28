# -*- coding: UTF-8 -*-

from flask import render_template, request, jsonify
from flask_restful import reqparse
from . import auth
from app.models.article import Article
from app import db
from werkzeug.utils import secure_filename
import os
import traceback
import datetime


MESSAGE = {
    "success": True,
    "message": "success",
    "res": {}
}

month = ['January', 'February', 'March', 'April', 'May',
         'June', 'July', 'August', 'September',
         'October', 'November', 'December']

@auth.route('/')
def index():
    return render_template('index.html', enable='')


@auth.route('/imgAdd', methods=['POST'])
def img_add():
    message = dict(MESSAGE)
    try:
        file = request.files['image']
        if file and allowed_image_file(file.filename):
            filename = secure_filename(file.filename)
            # print(os.path.pardir(__file__))
            path = os.path.join(os.path.dirname(__file__), os.path.pardir)
            t = str(datetime.datetime.now()) \
                .replace(":", "") \
                .replace("-", "") \
                .replace(":", "") \
                .replace(".", "") \
                .replace(" ", "")
            re_filename = t + "." + filename.split(".")[-1]
            file.save(os.path.join(os.path.abspath(path) + "/static/images", re_filename))
            message['url'] = "/images/" + re_filename
        else:
            message["success"] = False
            message["message"] = "不允许的图片格式！"
    except Exception as e:
        traceback.print_exc()
        print(e, flush=True)
        message["success"] = False
        message["message"] = "图片上传失败！"
    return jsonify(message)


@auth.route('/publish', methods=['POST'])
def publish():
    message = dict(MESSAGE)
    reqp = reqparse.RequestParser()
    reqp.add_argument("title", type=str, required=True, location=["json","form"])
    reqp.add_argument("tag", type=list, required=False, location=["json","form"])
    reqp.add_argument("privacy", type=bool, required=True, location=["json","form"])
    reqp.add_argument("article", type=str, required=True, location=["json","form"])
    args = reqp.parse_args()

    today = datetime.datetime.now()
    tags = request.json['tag']
    # print(request.json['tag'])

    if Article.query.filter_by(title=args["title"]).first() is not None:
        message["success"] = False
        message["message"] = "文章已存在！请换个标题。"
        return jsonify(message)

    article = Article(title=args["title"], tags="丨".join(tags),
                      article=args["article"], privacy=args["privacy"],
                      p_year=today.year, p_month=today.month, p_day=today.day)
    try:
        db.session.add(article)
        db.session.commit()
        # print(article.id)
        message["res"] = {"id": article.id}
    except Exception as e:
        print(e, flush=True)
        db.session.rollback()
        message["success"] = False
        message["message"] = "新建文章失败！"
    # print("--Log--: post article %s, %s" % (message["success"], message["message"]), flush=True)
    return jsonify(message)


@auth.route("/article/detail", methods=["POST"])
def detail():
    message = dict(MESSAGE)
    reqp = reqparse.RequestParser()
    reqp.add_argument("id", type=str, required=True, location=["json","form"])
    args = reqp.parse_args()

    article = Article.query.filter_by(id=args["id"]).first()
    if article is not None:
        res = {
            "title": article.title,
            "tags": article.tags.split("丨"),
            "detail": article.article,
            "datetime": month[article.p_month-1] + " " +
                        str(article.p_day) + ", " + str(article.p_year)
        }
        message["res"] = res
        return jsonify(message)
    else:
        message["success"] = False
        message["message"] = "找不到这篇文章"
        return jsonify(message), 404

@auth.route('/postlist', methods=['POST'])
def postlist():
    message = dict(MESSAGE)

    reqp = reqparse.RequestParser()
    reqp.add_argument("page", type=int, required=True, location=["json","form"])
    args = reqp.parse_args()

    article = Article.query.order_by(db.desc(Article.id)).\
        paginate(args["page"], per_page=5, error_out=False)
    article_list = []
    for item in article.items:
        article_list.append({
            "id": item.id,
            "datetime": month[item.p_month-1] + " " +
                        str(item.p_day) + ", " + str(item.p_year),
            "title": item.title,
            "tags": item.tags.split("丨")
        })
    message["res"] = {
        "postlist": article_list,
        "totalpage": article.pages
    }
    return jsonify(message)

# @auth.route('/imgDel', methods=['POST'])
# def img_del():
#     try:
#         message = dict(MESSAGE)
#         reqp = reqparse.RequestParser()
#         reqp.add_argument("filename", type=str, required=True, location=["json","form"])
#         args = reqp.parse_args()
#
#         filename = args['filename']
#         path = os.path.join(os.path.dirname(__file__), os.path.pardir)
#         path_name = os.path.join(os.path.abspath(path) + "/static", filename)
#         if path_name and os.path.exists(path_name):
#             os.remove(path_name)
#     except Exception as e:
#         traceback.print_exc()
#         print(e, flush=True)
#         message["success"] = False
#         message["message"] = "图片删除失败！"
#     return jsonify(message)


@auth.route('/archives')
def archives():
    return render_template('archives/index.html', enable='')
    

@auth.route('/2018/03/12/hello-world/')
def hello():
    return render_template('2018/03/12/hello-world/index.html', enable='')


@auth.app_errorhandler(404)
def not_found_error(error):
    return render_template('/error/404.html'),404


def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ["jpg", "jpeg", "bmp", "png", "tiff", "gif",
                                                  "raw", "svg"]
