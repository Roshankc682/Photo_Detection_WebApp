from uuid import uuid4
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import uuid
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from skimage.io import imread

def index(request):
    if request.method == "POST":
        img_details = request.FILES['image']
        fs = FileSystemStorage()
        filter_extension = os.path.splitext(img_details.name)
        fs = FileSystemStorage()
        img_details.name = str(uuid.uuid1())+filter_extension[1]
        fs.save(img_details.name, img_details)
        detect = "public/static/"+img_details.name
        image_path = "http://localhost:8000/media/" + img_details.name
        model = ResNet50(weights='imagenet')
        img_path  = detect
        img = imread(img_path)
        img = image.load_img(img_path, target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        preds = model.predict(x)
        data_predicted =  decode_predictions(preds, top=3)[0]
        return render(request,'index.html',{"first_name":(data_predicted[0][1]),
                                            "first_percent":str((data_predicted[0][2]*100))[slice(5)],
                                            "second_name":(data_predicted[1][1]),
                                            "second_percent":str((data_predicted[1][2]*100))[slice(5)],
                                            "third_name":(data_predicted[2][1]),
                                            "third_percent":str((data_predicted[2][2]*100))[slice(5)],
                                            "image_path":str(image_path),
                                            })
    else:
        return render(request,'index.html')
