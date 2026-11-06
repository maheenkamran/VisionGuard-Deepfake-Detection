from flask import Flask, render_template
from .config import Config
from .db import db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    with app.app_context():
        db.create_all()

    if Config.INFERENCE_MODE == "local":
        from .inference import Classifier
        app.classifier = Classifier(Config.MODEL_PATH, Config.THRESHOLD)

    from .routes import bp
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def too_big(e):
        return render_template("error.html", msg="File is larger than 5 MB.", code=413), 413

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", msg="Page not found.", code=404), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template("error.html", msg="Something went wrong on our side.", code=500), 500

    return app