# -*- coding: UTF-8 -*-

from app import app

if (__name__ == '__main__'):
    app.run('0.0.0.0', port=443, debug=False, threaded=True,
            ssl_context=('ssl/qsaltedfish/qsaltedfish.crt',
                         'ssl/qsaltedfish/qsaltedfish.key'))
