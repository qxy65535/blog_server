from app.models.article import Article
from app import  db

article = Article.query.order_by(db.desc(Article.id)).paginate(2,per_page=5,error_out=False)
for a in article.items:
    print(a.id, a.title, article.pages)


month = ['January', 'February', 'March', 'April', 'May',
         'June', 'July', 'August', 'September',
         'October', 'November', 'December']
print(month[0])