import urllib.request
from fastai.vision import *
from flask import Blueprint, render_template, request, Flask
import aiohttp

export_file_url = 'https://www.dropbox.com/s/v5unkltx0id5k1z/export.pkl?dl=1'
export_file_name = 'export.pkl'

def create_app():
    application = Flask(__name__)
    return application

app = create_app()

###################
classes = ['apples', 'oranges']

path = Path(__file__).parent

def download_file(url, dest):
    if dest.exists(): return
    with aiohttp.ClientSession() as session:
         with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

download_file(export_file_url, path / export_file_name)

# model
fruitLearner = load_learner(path, export_file_name)

def classify(bytes):
    img = open_image(BytesIO(bytes))
    pred_class, pred_idx, outputs = fruitLearner.predict(img)
    # print("fruit is: ", pred_class)
    return pred_class

###########################

@app.route('/', methods=['GET', 'POST'])
def home():
    # defaults.device = torch.device('cpu')

    if request.method == "POST":
        print("before getting image")
        bytes = request.files['file'].read()
        print(type(bytes))
        print("before making prediction")
        pred_class = classify(bytes)
        print("the fruit is: ", pred_class)
        return render_template("fruit.html", pred_class = pred_class)

    # elif request.method == "GET":
    #     imageURL = request.form.get("imageURL")
    #     print("image url: ", imageURL)
    #     urllib.request.urlretrieve(imageURL, './tmp/image.jpg')
    #     img = open_image('./tmp/image.jpg')
    #     pred_class, pred_idx, outputs = fruitLearner.predict(img)
    #     print("the fruit is: ", pred_class)
    #     return render_template("fruit.html", pred_class=pred_class)

    return render_template("home.html")

if __name__ == '__main__':
	app.debug = True
	app.run(host='127.0.0.1', port=8080, debug=True)