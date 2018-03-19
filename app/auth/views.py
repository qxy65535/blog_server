# -*- coding: UTF-8 -*-

from flask import render_template, request, jsonify
from flask_restful import reqparse
from . import auth
from werkzeug.utils import secure_filename
import os
import traceback
import datetime


MESSAGE = {
    "success": True,
    "message": "success",
    "res": {}
}


@auth.route('/')
def index():
    return render_template('index.html', enable='')


# @auth.route('/postlist', methods=['POST'])
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
            re_filename = t + "." + filename.split(".")[1]
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
    pass
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
