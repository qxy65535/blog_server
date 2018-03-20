#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

from app import db


class Article(db.Model):
    __tablename__ = "tb_articles"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), nullable=False)
    tags = db.Column(db.String(50), nullable=False)
    article = db.Column(db.Text, nullable=False)
    privacy = db.Column(db.Integer, nullable=False)
    p_year = db.Column(db.Integer, nullable=False)
    p_month = db.Column(db.Integer, nullable=False)
    p_day = db.Column(db.Integer, nullable=False)

