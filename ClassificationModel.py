#logloss function
import numpy as np
from sklearn.metrics import *
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
    def __init__(self,modfunc,metric,thresh=0.5):
        self.default_params=modfunc.get_params()
        self.model=modfunc
        self.metric=metric
        self.current={'params' : None
                      ,'predictions': None
                      , 'score' : None
                      , 'llscore' : None}
        self.best={'params' : None
                           , 'predictions': None
                           , 'score' : None
                   , 'llscore' : None}
        self.thresh=thresh
        self.description="A classification model, for use with ClassificationDataset"
        self.metricdict={"logloss": {'func': logloss
                                     ,'incflag' : -1.
                                     ,'binflag' : False},
                         "accuracy": {'func': accuracy_score
                                      ,'incflag':1.
                                      ,'binflag' : True},
                         "F1": {'func':f1_score
                                ,'incflag': 1.
                                ,'binflag' : True}}

    #store current and best models
    def fit(self,x,y,params=None,testflag=True):
        eflag=0
        if testflag is True:
            self.model.set_params(**self.default_params)
            if params is not None:
                self.model.set_params(**params)
        else:
            if metric in self.bpars.keys():
                params=self.best['params']
                self.model=self.modfunc
                self.model.set_params(**params)              
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
            if self.metricdict[self.metric]['binflag'] is True:
                pred_bin=(pred+0.5).astype(int)
                self.current['score']=self.metricdict[self.metric]['func'](y,pred_bin)
                self.current['llscore']=self.metricdict['logloss']['func'](y,pred_bin)                
            else:
                self.current['score']=self.metricdict[self.metric]['func'](y,pred)
                self.current['llscore']=self.metricdict['logloss']['func'](y,pred)
            print self.metric+" score: ",self.current['score']
            #updating best values
            if (self.best['score'] is None) or ((self.current['score']-self.best['score'])\
               *self.metricdict[self.metric]['incflag']>0.):
                self.best['score']=self.current['score']
                self.best['predictions']=self.current['predictions']
                self.best['params']=self.current['params']
                self.best['llscore']=self.current['llscore']
                print "new best "+self.metric+"!"
            else:
                print "current best "+self.metric+": ",self.best['score']
        else:
            return pred
