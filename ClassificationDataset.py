from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split,KFold
import pickle
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import SVC
from ClassificationModel import *
import matplotlib.pyplot as plt
from sklearn.metrics import *

class ClassificationDataset:
    def __init__(self,data,testdata,preprocessing=None,normed=False,reduced=False):
        self.normed=normed
        self.reduced=reduced
        self.split=False
        self.y_train=None
        self.y_test=None
        self.x_train=None
        self.x_test=None
        self.preprocessing=preprocessing
        if type(data) is list:
            #data and test data length must match
            if len(data) is not len(testdata):
                print "train and test data do not match in size"
            else:
                self.data=list()
                self.data_nvers=len(data)
                for i in range(self.data_nvers):
                    if i is 0:
                        self.target=data[0]['target']
                    try:
                        data[i]=data[i].drop('target',axis=1)
                    except:
                        print "no target for set ",i
                    self.data.append(data[i])
        else:
            self.target=data['target']
            self.data=list()
            self.data.append(data.drop('target',axis=1))
            self.data_nvers=1
        self.data_test=testdata
        self.description="A dataset class, to streamline the fitting process"
        self.author="Shawn Roberts"
        self.models={}
        self.ensemble_mod={}
        self.moddescriptions={"RF" : "Random Forest Classifier"
                      ,"KNN" : "K-Nearest Neighbors"
                      ,"ET" : "Extra Trees"
                      ,"SGD" : "Stochastic Gradient Descent"
                      ,'LR' : "Logistic Regression"
                      ,'Ada' : "AdaBoost Classifier"
                      ,'GBC' : "Gradient Boosting Classifier"
                      ,'SVC' : "Support Vector Machine Classifier"}
        self.moddict={"RF" : RandomForestClassifier()
                      ,"KNN" : KNeighborsClassifier()
                      ,"ET" : ExtraTreesClassifier()
                      ,"SGD" : SGDClassifier()
                      ,'LR' : LogisticRegression()
                      ,'Ada' : AdaBoostClassifier()
                      ,'GBC' : GradientBoostingClassifier()
                      ,'SVC' : SVC()}
        self.metricdict={"logloss": {'func': logloss
                                     ,'incflag' : -1.
                                     ,'binflag' : False},
                         "accuracy": {'func': accuracy_score
                                      ,'incflag':1.
                                      ,'binflag' : True},
                         "F1": {'func':f1_score
                                ,'incflag': 1.
                                ,'binflag' : True}}

    #return available models
    def get_available_models(self):
        return self.moddescriptions

    #clear saved stuff
    def flush(self):
        self.models={}
        self.split=False
        self.y_train=None
        self.y_test=None
        self.x_train=None
        self.x_test=None     
    
    #normalization
    def normalize(self,datanum,replace=False):
        if self.normed is False:
            data_stacked=sknormalize(self.data[datanum]+self.data_test[datanum])
            self.data[datanum]=data_stacked[:len(self.data[datanum])]
            self.data_test[datanum]=data_stacked[len(self.data[datanum]):]
            data_stacked=None
            self.normed=True
            #reset models
            self.flush()
        else:
            print "already normalized!"

    #dimensionality reduction with PCA
    def PCA(self,datanum,n_components=None,replace=False):
        if self.reduced is False:
            if self.normed is False:
                self.normalize()
            else:
                self.flush()
            pca=PCA(n_components=n_components)
            self.data[datanum]=pca.fit_transform(self.data[datanum],self.target)
            self.data_test[datanum]=pca.transform(self.data_test[datanum])
        else:
            print "already reduced with PCA"

    #split train-test
    def separate(self,test_size=0.3,random_state=1):
        if self.split is False:
            if type(self.data) is list:
                self.y_train=list()
                self.y_test=list()
                self.x_train=list()
                self.x_test=list()
                for i in range(self.data_nvers):
                    y_train_i, y_test_i, x_train_i, x_test_i\
                        = train_test_split(self.target
                                           , self.data[i]
                                           , test_size=test_size
                                           , random_state=random_state)
                    self.y_train.append(y_train_i)
                    self.y_test.append(y_test_i)
                    self.x_train.append(x_train_i)
                    self.x_test.append(x_test_i)     
            else:
                self.y_train, self.y_test, self.x_train, self.x_test\
                    = train_test_split(self.target
                                       , self.data
                                       , test_size=test_size
                                       , random_state=random_state)
            self.split=True
        else:
            print "already split!"

    #initialize a model with a custom name
    def init_model(self,strdict,metric,dsnum,thresh=0.5):
        modstring=strdict.keys()[0]
        if modstring not in self.models.keys():
            self.models[modstring]\
                =ClassificationModel(self.moddict[strdict[modstring]]
                                     ,metric
                                     ,dsnum
                                     ,thresh=thresh)
        else:
            print "model already exists!"

    #check if model exists
    def train_test(self,modstring,params=None,thresh=0.5):
        #doesn't exist yet
        if self.split is False:
            self.separate()
        if modstring not in self.models.keys():
            print "must initialize model first!"
        else:
            self.models[modstring].fit(self.x_train[self.models[modstring].dataset]
                                       ,self.y_train[self.models[modstring].dataset]
                                       ,params=params
                                       ,testflag=True)
            self.models[modstring].predict(self.x_test[self.models[modstring].dataset]
                                           ,self.y_test[self.models[modstring].dataset])           


    #train on the full dataset
    def train_full(self,modstring):
        if modstring in self.models.keys():
            self.models[modstring].fit(self.data[self.models[modstring].dataset]
                                       ,self.target
                                       ,testflag=False)
            return self.models[modstring].predict(self.data_test[self.models[modstring].dataset]) 
        else:
            "Error: model has not been tested yet!"


    #save method
    def save(self,filename):
        pickle.dump(self,file(filename,'w'))

    #delete a model
    def delete(self,modstr):
        self.models.pop(modstr,None)

    #get model scores
    def get_model_score(self,model):
        return {self.models[model].metric : self.models[model].best['score']}

    #get all models that have been run
    def get_models(self):
        return self.models.keys()
    
    #get all models and their scores of a given metric
    def get_metric_scores(self,metric):
        return_dict={}
        for key in self.models.keys():
            if metric is self.models[key].metric:
                return_dict[key]=self.models[key].best['score']
        return return_dict
    
    #get the best model given the score you want to optimize on
    def get_best_model(self,metric):
        best_mod=None
        best_score=None
        scoredict=self.get_metric_scores(metric)
        for key in scoredict.keys():
            if best_mod is None:
                best_mod=key
                best_score=scoredict[key]
            elif (scoredict[key]-best_score)\
            *self.models[key].metricdict[self.models[key].metric]['incflag']>0:
                best_mod=key
                best_score=scoredict[key]
        return best_mod

    #getting all true positives predictions
    def get_ones(self,model,current=False):
        if current is True:#use current
            return self.models[model].current['predictions'][np.array(self.y_test[self.models[model].dataset])==1]
        else:
            return self.models[model].best['predictions'][np.array(self.y_test[self.models[model].dataset])==1]

        
    #getting all genuine negatives predictions
    def get_zeros(self,model,current=False):
        if current is True:#use current
            return self.models[model].current['predictions'][np.array(self.y_test[self.models[model].dataset])==0]
        else:
            return self.models[model].best['predictions'][np.array(self.y_test[self.models[model].dataset])==0]

    
    #ensemble method# fit the optimal combination of each model
    #dsnum only applies to local fitting and should be chosen with care
    #it should only refer to a dataset that is normalized in some way
    def ensemble(self,method=None,metric='accuracy',n_neighbors=50,test=True,params=None,dsnum=0):
        #initialize new model; the ensemble model; internal to this class
        self.ensemble_mod['models']=self.models.keys()
        self.ensemble_mod['metric']=metric
        self.ensemble_mod['method']=method
        
        stacked_preds=np.zeros((len(self.models.keys()),len(self.y_test[0])))
        #weighting is done by the logloss score
        weights=np.ones(np.shape(stacked_preds))
        counter=0
        for key in self.models.keys():
            stacked_preds[counter]=self.models[key].best['predictions']
            weights[counter]=self.models[key].best['llscore']
            counter+=1
        weights=np.exp(-weights)
        #model averaging
        if method is None or method is 'average':
            self.ensemble_mod['predictions']=np.mean(stacked_preds,axis=0)
        #weighting average by the logloss score
        elif method is "weighted_average":
            self.ensemble_mod['predictions']=np.average(stacked_preds,axis=0,weights=weights)
        #vote and then average majority vote
        elif method is 'voted_average':
            votebool=stacked_preds>0.5
            votefrac=np.sum(votebool,axis=0)/float(len(weights))
            votedpreds=np.zeros(len(stacked_preds[0]))+0.5#if equal vote, then 0.5
            weighted=np.ones(np.shape(stacked_preds))
            weighted[np.logical_not(votebool)]=1.e-7
            votedpreds[votefrac>0.5]=np.average(stacked_preds,axis=0
                                                ,weights=weighted)[votefrac>0.5]
            weighted=np.ones(np.shape(stacked_preds))
            weighted[votebool]=1.e-7
            votedpreds[votefrac<0.5]=np.average(stacked_preds,axis=0
                                                ,weights=weighted)[votefrac<0.5]
            self.ensemble_mod['predictions']=votedpreds
        elif method is 'local':
            #print ClassifyDS.models['KNN'].model.kneighbors(ClassifyDS.x_test.iloc[11911].reshape(1,-1))[1][0]
            #print ClassifyDS.models['RF'].best_logloss['predictions'][neighbs]
            #train a new model on k nearest neighbors with the test set,
            #so we can locate the nearest, and then find the local logloss,
            #and smooth by n_neighbors
            kernmodel=KNeighborsClassifier(n_neighbors=n_neighbors,p=2)
            kernmodel.fit(self.x_test[dsnum],self.y_test[dsnum])
            #now we have our tree to locate neighbors
            #now, we want to iterate through the particles and find their local
            #logloss and determine which model to use
            if test is True:
                best_preds_test=np.zeros(len(self.y_test[dsnum]))
                #for test set
                for i in range(len(self.y_test[0])):
                    minlogloss=1.e7
                    neighbs=kernmodel.kneighbors(self.x_test[dsnum].iloc[i].reshape(1,-1))[1][0][1:]#throw out the actual observation
                    for key in self.models.keys():
                        logloss_loc=logloss(self.y_test[self.models[key].dataset].iloc[neighbs],self.models[key].best['predictions'][neighbs])
                        if logloss_loc<minlogloss:
                            minlogloss=logloss_loc
                            best_preds_test[i]=self.models[key].best['predictions'][i]
                self.ensemble_mod['predictions']=best_preds_test
            #running through the test data
            else:
                #predicting for each model
                prediction_dict={}
                for key in self.models.keys():
                    prediction_dict[key]=self.models[key].predict(self.data_test[self.models[key].dataset])
                best_preds_local=np.zeros(len(self.data_test[0]))
                for i in range(len(self.data_test[0])):
                    minlogloss=1.e7
                    neighbs=kernmodel.kneighbors(self.data_test[dsnum].iloc[i].reshape(1,-1))[1][0]
                    for key in self.models.keys():
                        logloss_loc=logloss(self.y_test[self.models[key].dataset].iloc[neighbs],self.models[key].best['predictions'][neighbs])
                        if logloss_loc<minlogloss:
                            minlogloss=logloss_loc
                            best_preds_local[i]=prediction_dict[key][i]
                self.ensemble_mod['predictions']=best_preds_local
        #averaging based on the local logloss score
        elif method is "local_average":
            kernmodel=KNeighborsClassifier(n_neighbors=n_neighbors,p=2)
            kernmodel.fit(self.x_test[dsnum],self.y_test[dsnum])
            if test is True:
                best_preds_test=np.zeros(len(self.y_test[dsnum]))
                #for test set
                logloss_i=np.zeros(len(self.models.keys()))
                preds_i=np.zeros(len(logloss_i))
                for i in range(len(self.y_test[dsnum])):
                    neighbs=kernmodel.kneighbors(self.x_test[dsnum].iloc[i].reshape(1,-1))[1][0][1:]#throw out the actual observation
                    counter=0
                    for key in self.models.keys():
                        logloss_i[counter]=logloss(self.y_test[dsnum].iloc[neighbs],self.models[key].best['predictions'][neighbs])
                        preds_i[counter]=self.models[key].best['predictions'][i]
                        counter+=1
                    logloss_i=np.exp(-logloss_i)
                    best_preds_test[i]=np.average(preds_i,weights=logloss_i)
                self.ensemble_mod['predictions']=best_preds_test
        elif method in self.moddict.keys():#stack results and throw to a new model
            #modelled predictions for withheld test set
            cv_preds=np.zeros((len(self.models.keys()),len(self.data[0])))
            #splitting in 3 fold set
            #and training each model
            folds=KFold(len(self.data[0]))
            counter=0
            for key in self.models.keys():
                for train_index,test_index in folds:
                    self.models[key].fit(self.data[self.models[key].dataset].iloc[train_index]
                                               ,self.target.iloc[train_index])
                    cv_preds[counter,test_index]=self.models[key].predict(self.data[self.models[key].dataset].iloc[test_index])
                counter+=1
            #now we split the predictions and train/test them
            cv_preds=np.transpose(cv_preds)
            y_train_int, y_test_int, x_train_int, x_test_int\
                = train_test_split(self.target
                                   , cv_preds
                                   , test_size=0.3
                                   , random_state=1)
            #create a new model instance
            model_int=ClassificationModel(self.moddict[method],metric,-1
                                          ,thresh=0.5)
            model_int.fit(x_train_int
                          ,y_train_int
                          ,params=params
                          ,testflag=True)
            model_int.predict(x_test_int
                              ,y_test_int)
            self.ensemble_mod['predictions']=model_int.best['predictions']
        if test is True:
            try:
                pred_bin=(self.ensemble_mod['predictions']+0.5).astype(int)
                if self.metricdict[metric]['binflag'] is True:
                    self.ensemble_mod['score']= self.metricdict[metric]['func'](np.array(self.y_test[0]),pred_bin)
                else:
                    self.ensemble_mod['score']= self.metricdict[metric]['func'](np.array(self.y_test[0]),self.ensemble_mod['predictions'])
                print "score: ",self.ensemble_mod['score']
            except:
                print "Unkown Method"


    #add the model as an input; prediction with separate models and K-fold CV
    def model_as_input(self,modstring):
        #modelled predictions for withheld test set
        cv_preds=np.zeros(len(self.data[0]))
        #splitting in 3 fold set
        folds=KFold(len(self.data[0]))
        #getting model parameters
        params=self.models[modstring].best['params']
        #iterating through splits and predicting
        for train_index,test_index in folds:
            self.models[modstring].fit(self.data[self.models[modstring].dataset].iloc[train_index]
                                       ,self.target.iloc[train_index]
                                       ,params=params
                                       ,testflag=True)
            cv_preds[test_index]=self.models[modstring].predict(self.data[self.models[modstring].dataset].iloc[test_index])
        #Now we add to the data and clear stored stuff
        self.data[modstring]=cv_preds
        self.flush()

    #scatterplot of ones and zeros
    def scatterplot(self,mod1,mod2,cur1=False,cur2=False):
        plt.plot(self.get_ones(mod1,current=cur1),self.get_ones(mod2,current=cur2),'.',label='1')
        plt.plot(self.get_zeros(mod1,current=cur1),self.get_zeros(mod2,current=cur2),'.',label='0')
        plt.plot(np.arange(11)/10.,np.arange(11)/10.)
        plt.legend()
        plt.show()
        
