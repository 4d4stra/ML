#logloss function
import numpy as np
def logloss(true,pred):
    pred[pred==1.]-=1.e-15
    pred[pred==0.]+=1.e-15
    return -np.sum(np.add(np.multiply(true,np.log(pred)),np.multiply(1-true,np.log(1-pred)))/len(true))

#positive likelihood ratio
def positive_lr(true,pred):
    tpr=len(pred[(pred>0.5) & (np.array(true)==1.)])/float(len(pred[np.array(true)==1.]))
    fpr=len(pred[(pred>0.5) & (np.array(true)==0.)])/float(len(pred[np.array(true)==0.]))
    try:
        plr=tpr/fpr
        return plr
    except ZeroDivisionError:
        return float('inf')

#positive likelihood ratio
def negative_lr(true,pred):
    tnr=len(pred[(pred<0.5) & (np.array(true)==0.)])/float(len(pred[np.array(true)==0.]))
    fnr=len(pred[(pred<0.5) & (np.array(true)==1.)])/float(len(pred[np.array(true)==1.]))
    try:
        nlr=fnr/tnr
        return nlr
    except ZeroDivisionError:
        return float('inf')
    
class ClassificationModel:
    def __init__(self,modfunc,thresh=0.5):
        self.default_params=modfunc.get_params()
        self.model=modfunc
        self.current={'params' : None
                      ,'predictions': None
                      , 'accsore' : None
                      , 'llscore' : None
                      , 'plrscore' : None
                      , 'nlrscore' : None}
        self.best_logloss={'params' : None
                           , 'predictions': None
                           , 'score' : 100.}
        self.best_accuracy={'params' : None
                            , 'predictions': None
                            ,'score' : 0.}
        self.best_positivelr={'params' : None
                            , 'predictions': None
                            ,'score' : 0.}
        self.best_negativelr={'params' : None
                            , 'predictions': None
                              ,'score' : 1.e10}
        self.bpars={'logloss' : None
                    ,'accuracy' : None
                    ,'positive_lr': None
                    ,'negative_lr': None}
        self.thresh=thresh
        self.description="A classification model, for use with"

    #store current and best models
    def fit(self,x,y,params=None,testflag=True,metric=None):
        eflag=0
        if testflag is True:
            self.model.set_params(**self.default_params)
            if params is not None:
                self.model.set_params(**params)
        elif metric is None:
            print "error, no metric defined"
            eflag=1
        else:
            if metric in self.bpars.keys():
                params=self.bpars[metric]
                self.model=self.modfunc
                self.model.set_params(**params)
            else:
                print "error, metric unknown"
                eflag=1                
        if eflag!=1:
            self.model.fit(x,y)

            
    #making predictions
    def predict(self,x,y=None):
        #predicting probability of target 1
        pred=self.model.predict_proba(x)[:,1]
        print "pred: ",len(pred)
        self.current['predictions']=pred
        self.current['params']=self.model.get_params()
        #calculating accuracy and logloss scores
        if y is not None:
            self.current['llscore']=logloss(y,pred)
            self.current['nlrscore']=negative_lr(y,pred)
            self.current['plrscore']=positive_lr(y,pred)
            self.current['accscore']=len(x[np.absolute(
                np.subtract(np.array(y),pred))<self.thresh])/float(len(x))
            print "logloss score: ",self.current['llscore']
            print "accuracy: ",self.current['accscore']
            print "positive likelihood ratio: ",self.current['plrscore']
            print "negative likelihood ratio: ",self.current['nlrscore']
            #updating best values
            #logloss score
            if self.current['llscore']<self.best_logloss['score']:
                self.best_logloss['score']=self.current['llscore']
                self.best_logloss['predictions']=self.current['predictions']
                self.best_logloss['params']=self.current['params']
                self.bpars['logloss']=self.current['params']
                print "new best llscore!"
            else:
                print "current best logloss: ",self.best_logloss['score']
            #accuracy
            if self.current['accscore']>self.best_accuracy['score']:
                self.best_accuracy['score']=self.current['accscore']
                self.best_accuracy['predictions']=self.current['predictions']
                self.best_accuracy['params']=self.current['params']
                self.bpars['accuracy']=self.current['params']
                print "new best accuracy!"
            else:
                print "current best accuracy: ",self.best_accuracy['score']
            #positive likelihood ratio
            if self.current['plrscore']>self.best_positivelr['score']:
                self.best_positivelr['score']=self.current['plrscore']
                self.best_positivelr['predictions']=self.current['predictions']
                self.best_positivelr['params']=self.current['params']
                self.bpars['positivelr']=self.current['params']
                print "new best positive likelihood!"
            else:
                print "current best positive likelihood: ",self.best_positivelr['score']
            if self.current['nlrscore']<self.best_negativelr['score']:
                self.best_negativelr['score']=self.current['nlrscore']
                self.best_negativelr['predictions']=self.current['predictions']
                self.best_negativelr['params']=self.current['params']
                self.bpars['negativelr']=self.current['params']
                print "new best negative likelihood!"
            else:
                print "current best negative likelihood: ",self.best_negativelr['score']
        else:
            return pred
