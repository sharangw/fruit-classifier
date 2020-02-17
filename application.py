import urllib.request
from fastai.vision import *
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse, PlainTextResponse
from flask import Blueprint, render_template, request, Flask

# app = Starlette()

def create_app():
    application = Flask(__name__)
    return application

app = create_app()

###################

classes = ['apples', 'oranges']
path = Path('.')
fruitsPath = Path('data')

# for c in classes:
#     print(c)
#     verify_images(path/c, delete=True, max_size=500)
#
#
# np.random.seed(42)
# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
#                                   ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#
# print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))

# train model
# fruitLearner = cnn_learner(data, models.resnet34, metrics=error_rate)
# fruitLearner.save('model1')

# classifier
# data2 = ImageDataBunch.create_from_ll(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
# learn = create_cnn(data2, models.resnet34)
# learn.load('model1')

# model
fruitLearner = load_learner(path, 'export.pkl')

def classify(bytes):
    img = open_image(BytesIO(bytes))
    pred_class, pred_idx, outputs = fruitLearner.predict(img)
    # print("fruit is: ", pred_class)
    return pred_class

# predict
#img = open_image(fruitsPath/'oranges'/'00000000.jpg')
# fruit = classify(img)
# print("FRUIT: ",fruit)

# pred_class, pred_idx, outputs = fruitLearner.predict(img)
# print("fruit is: ", pred_class)

###########################

@app.route('/', methods=['GET', 'POST'])
def home():
    defaults.device = torch.device('cpu')

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