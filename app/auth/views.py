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
import re
import base64
from io import BytesIO

from PIL import Image
import numpy as np

import torch
from app.util.net import net
import torch.nn.functional as F
from skimage.viewer import ImageViewer


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
        article = Article.query.filter_by(id=-1).first()
        message["res"] = article.article
        return jsonify(message), 404

@auth.route('/postlist', methods=['POST'])
def postlist():
    message = dict(MESSAGE)

    reqp = reqparse.RequestParser()
    reqp.add_argument("page", type=int, required=True, location=["json","form"])
    args = reqp.parse_args()

    article = Article.query.filter(Article.privacy==0).order_by(db.desc(Article.id)).\
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


@auth.route('/recognition', methods=['POST'])
def recognition():
    message = dict(MESSAGE)
    reqp = reqparse.RequestParser()
    reqp.add_argument("imageData", type=str, required=True, location=["json","form"])
    reqp.add_argument("x", type=int, required=True, location=["json","form"])
    reqp.add_argument("y", type=int, required=True, location=["json","form"])
    reqp.add_argument("endx", type=int, required=True, location=["json","form"])
    reqp.add_argument("endy", type=int, required=True, location=["json","form"])
    args = reqp.parse_args()

    # print(args["imageData"])
    image = torch.zeros(28, 28)
    mask = torch.zeros(28,28,dtype=torch.uint8)
    digit_image = base64_to_image(args["imageData"])
    # viewer = ImageViewer(np.asarray(image))
    # viewer.show()
    # image.show()
    x = args["x"]
    y = args["y"]
    endx = args["endx"]
    endy = args["endy"]
    # print(args["imageData"])
    digit_crop = digit_image.crop((x, y, endx, endy))
    # digit_crop.show()
    (crop_x, crop_y) = digit_crop.size
    print(digit_crop.size)
    if crop_x > crop_y:
        crop_y = int(crop_y / crop_x * 22)
        crop_x = 22

        start = int((28-crop_y)//2)
        mask[:, 3:25][start:start + crop_y, :] = 1
        print(crop_x, crop_y, 0)
    else:
        crop_x = int(crop_x / crop_y * 22)
        crop_y = 22

        start = int((28 - crop_x) / 2)

        mask[3:25, :][:, start:start + crop_x] = 1
        print(crop_x, crop_y, 1)

    digit_crop = digit_crop.resize((crop_x, crop_y), Image.ANTIALIAS)

    # digit_crop.show()
    im = 255 - np.asarray(digit_crop)
    # print(np.asarray(im))
    im_min, im_max = im.min(), im.max()  # 求最大最小值
    im = (im - im_min) / (im_max - im_min)  # (矩阵元素-最小值)/(最大值-最小值)
    im_t = torch.tensor(im).float()
    # im_t.masked_fill_(im_t.gt(0.9), 1)

    # print(mask)
    image.masked_scatter_(mask, im_t)

    viewer = ImageViewer(image.numpy())
    viewer.show()

    output = F.softmax(net(image.view(1, -1)), 1)
    print(output[0])
    _, predict = torch.max(output.data, 1)
    print(predict)
    message["message"] = {"predict":predict.item(),
                          "probability": dict(zip(range(10), [round(x.item(),3) for x in output[0]]))}

    # print(im.shape)
    # viewer = ImageViewer(im)
    # viewer.show()
    # im_t = torch.tensor(im).float()
    # im_t.masked_fill_(im_t.gt(0.5), 1)

    #     if file and allowed_image_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         # print(os.path.pardir(__file__))
    #         path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    #         t = str(datetime.datetime.now()) \
    #             .replace(":", "") \
    #             .replace("-", "") \
    #             .replace(":", "") \
    #             .replace(".", "") \
    #             .replace(" ", "")
    #         re_filename = t + "." + filename.split(".")[-1]
    #         file.save(os.path.join(os.path.abspath(path) + "/static/images", re_filename))
    #         message['url'] = "/images/" + re_filename
    #     else:
    #         message["success"] = False
    #         message["message"] = "不允许的图片格式！"

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

#
# @auth.route('/archives')
# def archives():
#     return render_template('archives/index.html', enable='')
#
#
# @auth.route('/2018/03/12/hello-world/')
# def hello():
#     return render_template('2018/03/12/hello-world/index.html', enable='')


@auth.app_errorhandler(404)
def not_found_error(error):
    return render_template('/error/404.html'),404


def allowed_image_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ["jpg", "jpeg", "bmp", "png", "tiff", "gif",
                                                  "raw", "svg"]


def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    # print(base64_data)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data).convert("L")
    return img