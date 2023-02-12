import tensorflow as tf
from flask import Flask, render_template, request, redirect
# redirect fonction permet de rediriger l'utilisateur vers une autre page#
import numpy as np
import cv2 #biblio traitement d'image
#permet de créer une
# application Flask#
app = Flask(__name__)
#une fonctionnalité de la bibliothèque Flask qui permet de
# définir une route pour la fonction#
@app.route('/', methods=['GET', 'POST'])
@app.route('/<filename>', methods=['GET', 'POST'])  #decorateur
def index(filename=''):
    # request contient les informations sur la requête HTTP en cours#
    # données envoyées par l'utilisateur#
    if request.method == 'POST':#POST,user a envoyé un fichier via un formulaire, recupere par request.files#
        f = request.files['file']
        f.save('static/uploads/' + "img.jpg")
        return redirect('/' + 'img.jpg')
    elif request.method=='GET': #GET pour afficher une page,user a accédé à la page via un lien#
        clothesPredicted = ''
        if filename != '':
            # read img from graphic file#
            img=cv2.imread('static/uploads/'+filename,cv2.IMREAD_GRAYSCALE)
            #resize an img#
            img=cv2.resize(img,(28,28),interpolation=cv2.INTER_LINEAR)
            # renvoie une image obtenue en appliquant l'opérateur not sur tous les bits de l'image#
            img=cv2.bitwise_not(img)
            print("test=" + filename)
            #gives a new shape to an array without changing the data#
            img=np.reshape(img, (1, 28, 28))
            #Loads a model saved via model.save()#
            model=tf.keras.models.load_model('static/models/Lemsyeh_Model.h5')
            #model du training#
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            #making predictions with keras#
            probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
            prediction=probability_model.predict(img)
            print(prediction)
            classnumber=np.argmax(prediction)#valeur max du tableau, get the index
            res=class_names[classnumber]
            print(res) #resultat du model#
        else:
            res=''
            filename=''
#permet de rendre un template HTML en utilisant les
#variables passées en paramètre#
        return render_template('index.html', imagePath=filename, pred=res)

app.run()
