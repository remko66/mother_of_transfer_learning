import os
from Mother_of_transfer_learning import application
from Mother_of_transfer_learning import fit
base_dir = './'
train_dir = base_dir+"images"
images = 764
epochs = 4
batch_size = 32

train_generator, test_generator = application().getGenerators(train_dir, batch_size=batch_size)
class_dictionary = train_generator.class_indices
x, y = train_generator.next()
cats = y.shape[1]
print("number of categories", cats)
print(application().AllAvailableModelsAsList())


# train all models to see what is the best one

def trainall():  # train all possible applications on a given set of images
    global train_generator, images, cats, test_generator
    fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator)
    for app in application().getall():
        app.saveloadpath = "models_desk/" + app.name
        fitnu.tensordir = base_dir + 'logs_desk/'
        if not os.path.isfile(app.saveloadpath):
            print("training " + app.name)
            model, lastacc, lastloss = fitnu.fitone(app)
            print(lastacc, lastloss)

trainall()
def trainone(name):
    global train_generator, images, cats, test_generator
    fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator)
    app = application.getApplicationByName(name)
    app.saveloadpath = "models_desk/" + app.name
    fitnu.tensordir = base_dir + 'logs_desk/'
    print("training " + app.name)
    model, lastacc, lastloss = fitnu.fitone(app)
    print(lastacc, lastloss)


# inference get the application, load the model
i = train_dir + "/n02085620-Chihuahua/n02085620_199.jpg"
app = application().getApplicationByName("vgg19", "models/vgg19_nu")
nr, full = app.inference(i)
print(application().predictionNumberToLabel(class_dictionary,nr))
print(nr, full)


def traincombo(apps, saveas='models/combination1',
               logdir=base_dir + "logs_desk/"):  # combine more then one application/model to one super transfer learning model...
    fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator)
    fitnu.tensordir = logdir
    fitnu.fitmodels(apps, saveas=saveas)
    if not os.path.isfile(app.saveloadpath):
        print("training " + app.name)
        model, lastacc, lastloss = fitnu.fitone(app)
        print(lastacc, lastloss)


# train a combination of applications/models and let the network decide wich features to use!
applist = []
app = application().getApplicationByName("vgg19", "models/vg19_weer",hastoload=True)
applist.append(app)
app = application().getApplicationByName("InceptionV3", "models/inceptionv3_weer",hastoload=True)
applist.append(app)
traincombo(applist)

# inference

i = train_dir + "/n02085620-Chihuahua/n02085620_199.jpg"
nr, full = application().inference_combined(applist, 'models/combination1', i)
print(application().predictionNumberToLabel(class_dictionary,nr))
print(nr, full)
