#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

from flask import current_app
from flask_login import UserMixin, AnonymousUserMixin
# from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from app import login_manager, db
# from app import app

# db = SQLAlchemy(app)

MESSAGE = {
"success":True,
"message":"success",
"res":{}
}

class Permission:
    ADMIN = 0xff

class User(UserMixin, db.Model):
    __tablename__ = 'tb_users'
    id = db.Column(db.String(10), primary_key=True)
    passwd = db.Column(db.String(128), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey('tb_roles.role_id'), nullable=False)

    def generate_auth_token(self, expiration=6000):
        s = Serializer(current_app.config['SECRET_KEY'], expiration)
        return s.dumps({'id':self.id})

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        return User.query.get(data['id'])

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.passwd = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.passwd, password)

    @login_manager.user_loader
    def load_user(userid):
        return User.query.get(str(userid))

    def can(self, permissions):
        return self.role is not None and (self.role.permission & permissions) == permissions

    def getRole(self):
        return self.role.role_name

    @staticmethod
    def check(stu, role):
        message = dict(MESSAGE)
        s = "学生学号"
        if role == 2:
            s = "教师教工号"
        if not "id" in stu:
            message["success"] = False
            message["message"] = "%s不能为空！" % s
        elif User.query.filter_by(id=stu["id"], role_id=role).first() is not None:
            message["success"] = False
            message["message"] = "%s：%s  已存在！" % (s, stu["id"])
        return message



class Role(db.Model):
    __tablename__ = 'tb_roles'
    role_id = db.Column(db.Integer, primary_key=True)
    role_name = db.Column(db.String(10), nullable=False)
    permission = db.Column(db.Integer, nullable=False)
    users = db.relationship('User', backref='role')

    @staticmethod
    def insert_roles():
        roles = {
            'student':Permission.STUDENT,
            'teacher':Permission.TEACHER,
            'admin':Permission.ADMIN
        }
        for r in roles:
            role = Role.query.filter_by(name=r).first()
            if role is None:
                role = Role(name=r)
            role.permissions = roles[r]
            db.session.add(role)
        db.session.commit()

    def get_role_id(self):
        return self.id

class AnonymousUser(AnonymousUserMixin):
    def can(self, permissions):
        return False
    def is_administrator(self):
        return False
login_manager.anonymous_user = AnonymousUser