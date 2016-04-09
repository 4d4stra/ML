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

class ClassificationDataset:
    def __init__(self,data,testdata,preprocessing=None,normed=False,reduced=False):
        self.normed=normed
        self.reduced=reduced
        self.split=False
        self.y_train=None
        self.y_test=None
        self.x_train=None
        self.x_test=None
        self.miniflag=False
        self.preprocessing=preprocessing
        self.target=data['target']
        self.data=data.drop('target',axis=1)
        self.data_test=testdata
        self.description="A dataset class, to keep track of shit"
        self.author="Shawn Roberts"
        self.models={}
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
    def normalize(self,replace=False):
        if self.normed is False:
            data_stacked=sknormalize(self.data+self.data_test)
            self.data=data_stacked[:len(self.data)]
            self.data_test=data_stacked[len(self.data):]
            data_stacked=None
            self.normed=True
            #reset models
            self.flush()
        else:
            print "already normalized!"

    #dimensionality reduction with PCA
    def PCA(self,n_components=None,replace=False):
        if self.reduced is False:
            if self.normed is False:
                self.normalize()
            else:
                self.flush()
            pca=PCA(n_components=n_components)
            self.data=pca.fit_transform(self.data,self.target)
            self.data_test=pca.transform(self.data_test)
        else:
            print "already reduced with PCA"

    #split train-test
    def separate(self,test_size=0.3,random_state=1):
        if self.split is False:
            self.y_train, self.y_test, self.x_train, self.x_test\
                = train_test_split(self.target
                                   , self.data
                                   , test_size=test_size
                                   , random_state=random_state)
            if test_size<=0.4:
                self.y_train_mini, self.y_valid, self.x_train_mini, self.x_valid\
                    = train_test_split(self.y_train
                                       , self.x_train
                                       , test_size=test_size/(1.-test_size)
                                       , random_state=random_state+1)
                self.miniflag=True
            else:
                self.miniflag=False
            self.split=True
        else:
            print "already split!"

    #initialize a model with a custom name
    def init_model(self,strdict,thresh=0.5):
        modstring=strdict.keys()[0]
        if modstring not in self.models.keys():
            self.models[modstring]\
                =ClassificationModel(self.moddict[strdict[modstring]]
                                     ,thresh=thresh)
        else:
            print "model already exists!"

    #check if model exists
    def train_test(self,modstring,params=None,thresh=0.5,mini=False):
        #doesn't exist yet
        if self.split is False:
            self.separate()
        if modstring not in self.models.keys():
            self.models[modstring]=ClassificationModel(self.moddict[modstring]
                                                       ,thresh=thresh)
        if mini is False:
            self.models[modstring].fit(self.x_train
                                       ,self.y_train
                                       ,params=params
                                       ,testflag=True)
        elif self.miniflag is True:
            self.models[modstring].fit(self.x_train_mini
                                       ,self.y_train_mini
                                       ,params=params
                                       ,testflag=True)
        else:
            print "test size too large to do mini test"
        self.models[modstring].predict(self.x_test
                                       ,self.y_test)           


    #train on the full dataset
    def train_full(self,modstring, metric='accuracy'):
        if modstring in self.models.keys():
            self.models[modstring].fit(self.data
                                       ,self.target
                                       ,testflag=False
                                       ,metric=metric)
            return self.models[modstring].predict(self.data_test) 
        else:
            "Error: model has not been tested yet!"


    #save method
    def save(self,filename):
        pickle.dump(self,file(filename,'w'))

    #delete a model
    def delete(self,modstr):
        self.models.pop(modstr,None)

    #get model scores
    def get_model_scores(self,model):
        return {'logloss' : self.models[model].best_logloss['score']
                ,'accuracy' : self.models[model].best_accuracy['score']
                ,'positivelr' : self.models[model].best_positivelr['score']
                ,'negativelr' : self.models[model].best_negativelr['score']}

    #get all models that have been run
    def get_models(self):
        return self.models.keys()
    
    #get all models and their scores of a given metric
    def get_metric_scores(self,metric):
        return_dict={}
        for key in self.models.keys():
            if metric is 'logloss':
                return_dict[key]=self.models[key].best_logloss['score']
            elif metric is 'accuracy':
                return_dict[key]=self.models[key].best_accuracy['score']
            elif metric is 'positivelr':
                return_dict[key]=self.models[key].best_positivelr['score']
            elif metric is 'negativelr':
                return_dict[key]=self.models[key].best_negativelr['score']
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
            elif ((metric is 'logloss') or (metric is 'negativelr')) and scoredict[key]<best_score:
                best_mod=key
                best_score=scoredict[key]
            elif ((metric is 'accuracy') or (metric is 'positivelr')) and scoredict[key]>best_score:
                best_mod=key
                best_score=scoredict[key]
        return best_mod

    #getting all true positives predictions
    def get_ones(self,model,metric=None):
        if metric is None or metric is 'current':#use current
            return self.models[model].current['predictions'][np.array(self.y_test)==1]
        elif metric is 'logloss':
            return self.models[model].best_logloss['predictions'][np.array(self.y_test)==1]
        elif metric is 'accuracy':
            return self.models[model].best_accuracy['predictions'][np.array(self.y_test)==1]
        elif metric is 'positivelr':
            return self.models[model].best_positivelr['predictions'][np.array(self.y_test)==1]
        elif metric is 'negativelr':
            return self.models[model].best_negativelr['predictions'][np.array(self.y_test)==1]

        
    #getting all genuine negatives predictions
    def get_zeros(self,model,metric=None):
        if metric is None or metric is 'current':#use current
            return self.models[model].current['predictions'][np.array(self.y_test)==0]
        elif metric is 'logloss':
            return self.models[model].best_logloss['predictions'][np.array(self.y_test)==0]
        elif metric is 'accuracy':
            return self.models[model].best_accuracy['predictions'][np.array(self.y_test)==0]
        elif metric is 'positivelr':
            return self.models[model].best_positivelr['predictions'][np.array(self.y_test)==0]
        elif metric is 'negativelr':
            return self.models[model].best_negativelr['predictions'][np.array(self.y_test)==0]

    
    #ensemble method# fit the optimal combination of each model
    def ensemble(self,method=None,n_neighbors=50,test=True,params=None):
        stacked_preds=np.zeros((len(self.models.keys()),len(self.y_test)))
        weights=np.ones(np.shape(stacked_preds))
        counter=0
        for key in self.models.keys():
            stacked_preds[counter]=self.models[key].best_logloss['predictions']
            weights[counter]=self.models[key].best_logloss['score']
            counter+=1
        weights=np.exp(-weights)
        if method is None or method is 'average':#averaging
            return logloss(np.array(self.y_test),np.mean(stacked_preds,axis=0))
        elif method is "weighted_average":
            return logloss(np.array(self.y_test)
                           ,np.average(stacked_preds,axis=0,weights=weights))
        elif method is 'max_confidence':#taking the maximally confident prediction
            max_conf=np.amax(np.absolute(stacked_preds-0.5),axis=0)
            max_conf_pred=np.zeros(len(max_conf))
            for i in range(len(max_conf)):
                if type(stacked_preds[:,i][np.absolute(stacked_preds[:,i]-0.5)==max_conf[i]]) is float:
                    max_conf_pred[i]=stacked_preds[:,i][np.absolute(stacked_preds[:,i]-0.5)==max_conf[i]]
                else:
                    max_conf_pred[i]=stacked_preds[:,i][np.absolute(stacked_preds[:,i]-0.5)==max_conf[i]][0]
            return logloss(np.array(self.y_test),max_conf_pred)
        elif method is 'min_confidence':#taking the maximally confident prediction
            min_conf=np.amin(np.absolute(stacked_preds-0.5),axis=0)
            min_conf_pred=np.zeros(len(min_conf))
            for i in range(len(min_conf)):
                if type(stacked_preds[:,i][np.absolute(stacked_preds[:,i]-0.5)==min_conf[i]]) is float:
                    min_conf_pred[i]=stacked_preds[:,i][np.absolute(stacked_preds[:,i]-0.5)==min_conf[i]]
                else:
                    min_conf_pred[i]=stacked_preds[:,i][np.absolute(stacked_preds[:,i]-0.5)==min_conf[i]][0]
            return logloss(np.array(self.y_test),min_conf_pred)
        elif method is 'voted_average':#vote and then average
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
            return logloss(np.array(self.y_test),votedpreds)
        elif method is 'local':
            #print ClassifyDS.models['KNN'].model.kneighbors(ClassifyDS.x_test.iloc[11911].reshape(1,-1))[1][0]
            #print ClassifyDS.models['RF'].best_logloss['predictions'][neighbs]
            #train a new model on k nearest neighbors with the test set,
            #so we can locate the nearest, and then find the local logloss,
            #and smooth by n_neighbors
            kernmodel=KNeighborsClassifier(n_neighbors=n_neighbors,p=2)
            kernmodel.fit(self.x_test,self.y_test)
            #now we have our tree to locate neighbors
            #now, we want to iterate through the particles and find their local
            #logloss and determine which model to use
            if test is True:
                best_preds_test=np.zeros(len(self.y_test))
                #for test set
                for i in range(len(self.y_test)):
                    minlogloss=1.e7
                    neighbs=kernmodel.kneighbors(self.x_test.iloc[i].reshape(1,-1))[1][0][1:]#throw out the actual observation
                    for key in self.models.keys():
                        logloss_loc=logloss(self.y_test.iloc[neighbs],self.models[key].best_logloss['predictions'][neighbs])
                        if logloss_loc<minlogloss:
                            minlogloss=logloss_loc
                            best_preds_test[i]=self.models[key].best_logloss['predictions'][i]
                return logloss(np.array(self.y_test),best_preds_test)

            else:
                #predicting for each model
                prediction_dict={}
                for key in self.models.keys():
                    prediction_dict[key]=self.models[key].predict(self.data_test)
                print prediction_dict.keys()
                print prediction_dict['RF']
                print len(self.models['RF'].current['predictions'])
                print len(self.models['RF'].best_logloss['predictions'])
                best_preds_local=np.zeros(len(self.data_test))
                for i in range(len(self.data_test)):
                    minlogloss=1.e7
                    neighbs=kernmodel.kneighbors(self.data_test.iloc[i].reshape(1,-1))[1][0]
                    for key in self.models.keys():
                        logloss_loc=logloss(self.y_test.iloc[neighbs],self.models[key].best_logloss['predictions'][neighbs])
                        if logloss_loc<minlogloss:
                            minlogloss=logloss_loc
                            best_preds_local[i]=prediction_dict[key][i]
                return best_preds_local
        elif method is "local_average":
            kernmodel=KNeighborsClassifier(n_neighbors=n_neighbors,p=2)
            kernmodel.fit(self.x_test,self.y_test)
            if test is True:
                best_preds_test=np.zeros(len(self.y_test))
                #for test set
                logloss_i=np.zeros(len(self.models.keys()))
                preds_i=np.zeros(len(logloss_i))
                for i in range(len(self.y_test)):
                    neighbs=kernmodel.kneighbors(self.x_test.iloc[i].reshape(1,-1))[1][0][1:]#throw out the actual observation
                    counter=0
                    for key in self.models.keys():
                        logloss_i[counter]=logloss(self.y_test.iloc[neighbs],self.models[key].best_logloss['predictions'][neighbs])
                        preds_i[counter]=self.models[key].best_logloss['predictions'][i]
                        counter+=1
                    logloss_i=np.exp(-logloss_i)
                    best_preds_test[i]=np.average(preds_i,weights=logloss_i)
                return logloss(np.array(self.y_test),best_preds_test)
        elif method in self.moddict.keys():#stack results and throw to a new model
            #modelled predictions for withheld test set
            cv_preds=np.zeros((len(self.models.keys()),len(self.data)))
            #splitting in 3 fold set
            #and training each model
            folds=KFold(len(self.data))
            counter=0
            for key in self.models.keys():
                for train_index,test_index in folds:
                    self.models[key].fit(self.data.iloc[train_index]
                                               ,self.target.iloc[train_index]
                                               ,metric='logloss')
                    cv_preds[counter,test_index]=self.models[key].predict(self.data.iloc[test_index])
                counter+=1
            #now we split the predictions and train/test them
            cv_preds=np.transpose(cv_preds)
            y_train_int, y_test_int, x_train_int, x_test_int\
                = train_test_split(self.target
                                   , cv_preds
                                   , test_size=0.3
                                   , random_state=1)
            #create a new model instance
            model_int=ClassificationModel(self.moddict[method]
                                          ,thresh=0.5)
            model_int.fit(x_train_int
                          ,y_train_int
                          ,params=params
                          ,testflag=True)
            model_int.predict(x_test_int
                              ,y_test_int)   
        else:
            print "Unknown method."


    #add the model as an input; prediction with separate models and K-fold CV
    def model_as_input(self,modstring,metric):
        #modelled predictions for withheld test set
        cv_preds=np.zeros(len(self.data))
        #splitting in 3 fold set
        folds=KFold(len(self.data))
        #getting model parameters
        params=self.models[modstring].bpars[metric]
        #iterating through splits and predicting
        for train_index,test_index in folds:
            self.models[modstring].fit(self.data.iloc[train_index]
                                       ,self.target.iloc[train_index]
                                       ,params=params
                                       ,testflag=True)
            cv_preds[test_index]=self.models[modstring].predict(self.data.iloc[test_index])
        #Now we add to the data and clear stored stuff
        self.data[modstring]=cv_preds
        self.flush()

    #scatterplot of ones and zeros
    def scatterplot(self,mod1,mod2):
        plt.plot(self.get_ones(mod1,'logloss'),self.get_ones(mod2,'logloss'),'.',label='1')
        plt.plot(self.get_zeros(mod1,'logloss'),self.get_zeros(mod2,'logloss'),'.',label='0')
        plt.plot(np.arange(11)/10.,np.arange(11)/10.)
        plt.legend()
        plt.show()
        
