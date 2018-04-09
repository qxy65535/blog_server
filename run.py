# -*- coding: UTF-8 -*-

from app import app

if (__name__ == '__main__'):
    app.run('0.0.0.0', port=80, debug=False, threaded=True,
            ssl_context=('/home/ubuntu/ssl/qsaltedfish/1_qsaltedfish.cn_bundle.crt',
                         '/home/ubuntu/ssl/qsaltedfish/2_qsaltedfish.cn.key'))
