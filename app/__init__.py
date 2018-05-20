#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

from flask import Flask
from flask_login import LoginManager
from flask_cors import CORS
from flask_sslify import SSLify


login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_message = u"请登录您的账户!"
login_manager.login_view = "view.logintest"

app = Flask(__name__, static_url_path="")
# sslify = SSLify(app)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:software@139.199.71.90/blog'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:software@127.0.0.1/blog'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:q87268868@localhost/blog'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
UPLOAD_FOLDER = 'img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
login_manager.init_app(app)

from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)
# from .database.db import Database
# db = Database()
from .auth import auth
# from .api import api as api_v1_0
app.register_blueprint(auth)
