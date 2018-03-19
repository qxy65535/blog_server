#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

from app import db

class Article(db.Model):
    __tablename__ = "tb_course_status"
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(10), nullable=False)

