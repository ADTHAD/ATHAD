from flask import Flask, render_template, jsonify, redirect, request
from heart import predict_stuff
from liver import predict_liver_stuff as liver_prediction
from diabetes import predict_diabetes_stuff


app = Flask(__name__)

@app.route("/heart.html", methods=['GET'])
def heart():
    return render_template("heart.html")
@app.route("/", methods=['GET'])
def index():
    return render_template("index.html") 
@app.route("/pneumonia.html", methods=['GET'])
def pneumonia():
    return render_template("pneumonia.html")
@app.route("/liver.html", methods=['GET'])
def liver():
    return render_template("liver.html")
@app.route("/diabetes.html", methods=['GET'])
def diabetes():
    return render_template("diabetes.html")   
@app.route("/about.html", methods=['GET'])
def about():
    return render_template("about.html")


@app.route('/heart-user-data', methods=['POST'])
def heart_predict():
    result=predict_stuff()
    if result:
        return render_template("heart.html", pred=result)
    else:
        return render_template("error.html")

@app.route('/liver-user-data', methods=['POST'])
def liver_predict():
    result=liver_prediction()
    if result:
        return render_template("liver.html", liver_pred=result)
    else:
        return render_template("liver_error.html")

@app.route('/diabetes-user-data', methods=['POST'])
def diabetes_predict():
    result=predict_diabetes_stuff()
    if result:
        return render_template("diabetes.html", diabetes_pred=result[0], accuracy=str(result[1]))
    else:
        return render_template("error_diabetes.html")    


if __name__ == "__main__":
    app.run()
