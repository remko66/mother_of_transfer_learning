import os
from Mother_of_transfer_learning import application
from Mother_of_transfer_learning import fit

base_dir = './'
train_dir = base_dir+"images"
images = 764
epochs = 1
batch_size = 32

#what are we going to do?
trainall=False
evaluatesome=False
inference=False
traincombo=True
Evaluatecombo=True

#get the generator on a dir structure
train_generator, test_generator = application().getGenerators(train_dir, batch_size=batch_size)
class_dictionary = train_generator.class_indices
x, y = train_generator.next()
cats = y.shape[1]
print("number of categories", cats)
print(application().AllAvailableModelsAsList())


# train all models to see what is the best one
if trainall:
    app=application().getApplicationByName("DenseNet201","models/DenseNet201",hastoload=True)
    acc,loss=app.Evaluate_one(test_generator)
    print(acc,loss)



def evaluate_one(modelname,savepath,testgen):
    app=application().getApplicationByName(modelname,saveloadpath=savepath,hastoload=True)
    return app.Evaluate_one_saved(testgen,savepath)

def evaluate_combined(applist,savepath,testgen):
    app=application().Evaluate_combined(testgen, applist,savepath)
    return app

def trainall():  # train all possible applications on a given set of images
    global train_generator, images, cats, test_generator
    fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator,epochs=epochs)
    for app in application().getall():
        app.saveloadpath = "models_desk/" + app.name+"_v2"
        fitnu.tensordir = base_dir + 'logs_desk/'
        if not os.path.isfile(app.saveloadpath):
            if not os.path.isfile(app.saveloadpath+".h5"):
                print("training " + app.name)
                model, lastacc, lastloss = fitnu.fitone(app)
                print(lastacc, lastloss)


def trainone(name):
    global train_generator, images, cats, test_generator
    fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator)
    app = application.getApplicationByName(name)
    app.saveloadpath = "models_desk/" + app.name
    fitnu.tensordir = base_dir + 'logs_desk/'
    print("training " + app.name)
    model, lastacc, lastloss = fitnu.fitone(app)
    print(lastacc, lastloss)



#train combination of two or more models
def traincombo(apps, saveas='models/combination1',epochs=epochs,
               logdir=base_dir + "logs_combo/",combtype="concatenate"):  # combine more then one application/model to one super transfer learning model...
    fitnu = fit(train_generator, images, cats, datageneretor_test=test_generator,epochs=epochs)
    fitnu.tensordir = "logs_combo'"+combtype
    fitnu.fitmodels(apps, saveas=saveas,combtype=combtype)




# train a combination of applications/models and let the network decide wich features to use!
if traincombo:
    applist = []
    app = application().getApplicationByName("InceptionV3")
    applist.append(app)
    app = application().getApplicationByName("DenseNet201")
    applist.append(app)
    traincombo(applist,saveas="models/combDenseNet201_InceptionV3_concat",combtype='concatenate')
   # traincombo(applist, saveas="models/combDenseNet201_InceptionV3_add", combtype='add')
   # traincombo(applist, saveas="models/combDenseNet201_InceptionV3_average", combtype='average')
   # traincombo(applist, saveas="models/combDenseNet201_InceptionV3_multiply", combtype='multiply')


#evaluate some single models
if evaluatesome:
    print("inception",evaluate_one("InceptionV3","models_desk/InceptionV3_v2",test_generator))
    print("densenet",evaluate_one("DenseNet201","models_desk/DenseNet201_v2",test_generator))


#evaluate model combo's
if Evaluatecombo:
    applist = []
    app = application().getApplicationByName("InceptionV3", "models_desk/InceptionV3_v2", hastoload=True)
    applist.append(app)
    app = application().getApplicationByName("DenseNet201", "models_desk/DenseNet201_v2", hastoload=True)
    applist.append(app)
    print("eval combined concat",evaluate_combined(applist,"models/combDenseNet201_InceptionV3_concat",test_generator))
    #print("eval combined add", evaluate_combined(applist, "models/combDenseNet201_InceptionV3_add", test_generator))
    #print("eval combined average", evaluate_combined(applist, "models/combDenseNet201_InceptionV3_average", test_generator))
    #print("eval combined multiply", evaluate_combined(applist, "models/combDenseNet201_InceptionV3_multiply", test_generator))


# inference
if inference:
    i = train_dir + "/n02085620-Chihuahua/n02085620_199.jpg"
    nr, full = application().inference_combined(applist, 'models/combination1', i)
    print(application().predictionNumberToLabel(class_dictionary,nr))
    print(nr, full)
