from package.database_models import *


class DatabaseOperator:

    @staticmethod
    def insert(model, kwargs, app=None):
        if app:
            with app.app_context():
                db.session.add(model(**kwargs))
                db.session.commit()
        else:
            db.session.add(model(**kwargs))
            db.session.commit()

    @staticmethod
    def update(model, filter_kwargs, new_kwargs, app=None):
        if app:
            with app.app_context():
                data = DatabaseOperator.select_one(model, filter_kwargs)
                for k, v in new_kwargs.items():
                    setattr(data, k, v)
                db.session.commit()
        else:
            data = DatabaseOperator.select_one(model, filter_kwargs)
            for k, v in new_kwargs.items():
                setattr(data, k, v)
            db.session.commit()

    @staticmethod
    def select_one(model, kwargs, app=None):
        if app:
            with app.app_context():
                return db.session.query(model).filter_by(**kwargs).first()
        else:
            return db.session.query(model).filter_by(**kwargs).first()

    @staticmethod
    def select_all(model, app=None):
        if app:
            with app.app_context():
                return db.session.query(model).all()
        else:
            return db.session.query(model).all()
