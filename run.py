# -*- coding: UTF-8 -*-

from app import app

if (__name__ == '__main__'):
    app.run('0.0.0.0', port=80, debug=False, threaded=True)
    # app.run('0.0.0.0', port=80, debug=False, threaded=True,
    #         ssl_context=('D:/1work/ssl/qsaltedfish/Nginx/qsaltedfish.crt',
    #                      'D:/1work/ssl/qsaltedfish/Nginx/qsaltedfish.key'))
