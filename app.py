from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


@app.route("/")
def index():






    return render_template("index.html")





if __name__== "__name__":
    app.run(debug=False,host=  "0.0.0.0")