import os
import keras
import numpy as np
from keras.layers import GlobalAveragePooling2D, Dense, concatenate
from keras.models import Model
from Mother_of_transfer_learning.application import application


class fit:
    def __init__(self,datagenerator_train,nrimages,nrcats,datageneretor_test=None,batch_size=32,epochs=5,use_tensorboard=True,tensordir='log/'):

        self.gen_train=datagenerator_train
        self.gen_test=datageneretor_test
        self.batchsize=batch_size
        self.epochs=epochs
        self.nrimages=nrimages
        self.nrcats=nrcats
        self.use_bord=use_tensorboard
        self.tensordir=tensordir

    def maketensorboard(self,model,dir):
        self.tensorboard = keras.callbacks.TensorBoard(
            log_dir=dir,
            histogram_freq=0,
            batch_size=self.batchsize,
            write_graph=True,
            write_grads=True
        )
        if not os.path.exists(self.tensordir):
            os.makedirs(self.tensordir)
        self.tensorboard.set_model(model)

    def named_logs(self,model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def fitone(self,app,save=True):
        instance = app.getinstance()
        nowdir=self.tensordir+app.name
        x = instance.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.nrcats, activation='relu')(x)
        x = Dense(self.nrcats, activation='relu')(x)
        preds = Dense(self.nrcats, activation='softmax')(x)
        model = Model(inputs=app.input_tensor, outputs=preds)

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.maketensorboard( model,nowdir)
        steps = round(self.nrimages /self.batchsize)
        batch_id = 1
        for e in range(self.epochs):
            for a in range(steps):
                x, y = self.gen_train.next()
                x = app.resizeAndPreprocess(x)
                res = model.train_on_batch(x, y)
                self.tensorboard.on_epoch_end(batch_id, self.named_logs(model, res))
                if a % 50 == 0:
                    print("epoch", e + 1, "/", self.epochs, ' ', a + 1, '/', steps, '-->', res)
                batch_id += 1
        if not self.gen_test==None:
            x, y = self.gen_test.next()
            x = app.resizeAndPreprocess(x)
            history = model.evaluate(x, y, verbose=1)
            print(history)
            lastacc = history[1]
            lastloss = history[0]
        else:
            lastacc,lastloss=0
        if save:
            p=app.saveloadpath
            if not "." in p:
                p+=app.defaultext
            model.save(app.saveloadpath)
        return model,lastacc, lastloss


    def fitByName(self,name,save=True):
        app=application().getApplicationByName(name)
        return self.fitone(app,save)





    def fitmodels(self,applist,saveas="models/combined"):
        pre=[]
        out=[]
        input=[]
        for model in applist:
            instance=model.getinstance(cancreate=False)
            pre.append(model.preprocces_func)
            input.append(instance.input)
            x=instance.output
            out.append(x)
        x=concatenate(out)
        x = Dense(self.nrcats, activation='relu')(x)
        preds = Dense(self.nrcats, activation='softmax')(x)
        model = Model(inputs=input, outputs=preds)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.maketensorboard("Combined", model,self.tensordir+"combined")
        steps = round(self.nrimages / self.batchsize)
        batch_id = 1
        for e in range(self.epochs):
            for a in range(steps):
                x, y = self.gen_train.next()
                x = application().makeinputmulti(pre,x)
                res = model.train_on_batch(x, y)
                self.tensorboard.on_epoch_end(batch_id, self.named_logs(model, res))
                if a % 50 == 0:
                    print("epoch", e + 1, "/", self.epochs, ' ', a + 1, '/', steps, '-->', res)
                batch_id += 1
        if not self.gen_test==None:
            x, y = self.gen_test.next()
            x = application().makeinputmulti(pre, x)
            history = model.evaluate(x, y, verbose=1)
            print(history)
            lastacc = history[1]
            lastloss = history[0]
        else:
            lastloss,lastacc=0
        if not saveas=="":
            model.save(saveas)
        return lastacc, lastloss

