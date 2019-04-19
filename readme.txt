# Mother of all transfer learning

Version 0.5 (work in progress, the single applications should work fine. The combined is not fully tested yet).

## tools used

Its keras 2.2 and open cv 4

## What is it?
The library is meant to take all the work involved in Tranfer learning for image recognition out of your hands.

If you have an image collection according to this organization, training a network is a breeze.
If you don't know which one to choose, train them all(Everybody needs to sleep right...so can take some time)

You can even combine some trained networks into a 'meta' network, where the system decides which one to use for feature Extraction!


The image dir should look like this for training:

images
--label1
-----imglabel1
-----imglabel1
-----imglabel1
--label2
-----imglabel2
-----imglabel2
-----imglabel2

If you ever have done anything with image recognition you will recognize this structure. Else google how to put images in dirs for image recognition. Almost everybody uses this structure.


## How to use it?

from Mother_of_transfer_learning import application
from Mother_of_transfer_learning import fit

Now set some basic params:

train_dir = "images"
images = 14421 #the total number of images
epochs = 4 #number of epochs (a good rule of thumb is 60000/nrofimages. With minimal 500-1000 images
batch_size = 32 # if you don't know what it is, 32 will mostly do just fine

Application is the word keras uses for pretrained models. So i kind of copy it.

Now get your datagenerator :
train_generator, test_generator = application().getGenerators(train_dir, batch_size=batch_size)
class_dictionary = train_generator.class_indices

x, y = train_generator.next()
cats = y.shape[1] #number of categories

To train a network for transfer learning:
name="VGG19" #just a sample, DenseNet201 should be much better...anyway...try them all

 fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator)
 app = application.getApplicationByName(name)
 app.saveloadpath = "models/" + app.name + "_version1"
 fit.tensordir =  'log/' #not obligatory, you can even turn tensorboard loggin on when instantiating fit
 print("training " + app.name)
 model, lastacc, lastloss = fitnu.fitone(app)
 print(lastacc, lastloss)

 Done! you have a trained model!!!!

 Inference:

i = train_dir + "/n02085620-Chihuahua/n02085620_199.jpg" #a picture we want to predict
app = application().getApplicationByName("vgg19", "models/vgg19_version1") #get the application and the saved model
nr, full = app.inference(i)
print(application().predictionNumberToLabel(class_dictionary,nr))


Thats it!

See the code in sample.py You can also find there how to train a network on a combination of earlier trained applications!
I include the dog breed dataset from kaggle.com. (guys, if you have an issue with that let me know, i directly delete it. I know everybody can download it also at your site(great fan btw).


So basically 2 classes:

1. Application, does everything around the model (for example Densenet)
2. fit (fitting name for a fitting class)

To get all applications/models currently supported run:

print(application().AllAvailableModelsAsList())

Under the hood functionality:

Every image if resized and preprocessed as to the liking of the model you use. Some models where trained on different sizes of pictures.

Check getall in application and you will see what i mean. Feel free to add more models!

If i have some time will document the classes so you don't have to spy in the code(but it should be simple enough).

If you need any coding done (specially neural nets or data science stuff, but boring andoid apps are oke also): please let me know:

Whatsapp: +628158370088 (English, dutch or a little bit of bahasa indonesia)
Remko66@gmail.com

You can not use it from the command line, it is meant as a package to be included in your project.