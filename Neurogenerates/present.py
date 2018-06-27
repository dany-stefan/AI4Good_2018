# run: python present.py

# coding: utf-8

# # AI For Social Good
# ## NeuroGenerates
# ### An AI Triage Tool for your Health 
# Carolyne, Jules, Mary, Rez, Sam

# In[1]:


import pandas as pd
import numpy as np


# # Sanitize the Data
# We sanitized the Data by removing duplicates with the uni_name function and by replacing empty ages by an average of age, which is 32

# In[2]:


def cleanTrackable_type(x):
    if x in ['Condition','Symptom','Treatment']:
        clean = True
    else:
        clean = False
    return clean


# In[3]:


df = pd.read_csv('fd-export.csv')
df = df[[cleanTrackable_type(i) for i in df['trackable_type']]]
df['trackable_name'] = [i.lower() for i in df['trackable_name']]
df['age'] = df.age.replace(0.0,np.nan)
df.head()


# In[4]:


df['age'] = df.age.replace(np.nan,32)
df['age'] = df['age']/50.0
df['sex'] = df.sex.replace(np.nan,'doesnt_say')
df=df.dropna(subset = ['trackable_name'])


# In[5]:


def uni_name(s):
  if s=='neck and shoulder pain': return 'neck pain'
  elif s=='migraines': return 'migraine'
  elif s=='dislocations': return 'subluxation'
  elif s=='right hip pain': return 'hip pain'
  elif s=='left hip pain': return 'hip pain'
  elif s=='right shoulder pain': return 'shoulder pain'
  elif s=='left shoulder pain': return 'shoulder pain'
  elif s=='physical fatigue': return 'fatigue'
  elif s=='tiredness': return 'fatigue'
  elif s=='muscle ache': return 'fatigue'
  elif s=='body aches': return 'fatigue'
  elif s=='body aching': return 'fatigue'
  elif s=='pain': return 'fatigue'
  elif s=='exhaustion': return 'fatigue'
  elif s=='muscle pain': return 'fatigue'
  elif s=='fatigue and tiredness': return 'fatigue'
  elif s=='headaches': return 'headache'
  elif s=='vomiting': return 'nausea'
  elif s=='social anxiety': return 'anxiety'
  elif s=='stress': return 'anxiety'
  elif s=='lightheadedness': return 'dizziness'
  elif s=='dizzyness': return 'dizziness'
  elif s=='middle back pain': return 'back pain'
  elif s=='body aching': return 'back pain'
  elif s=='lower abdomen pain': return 'stomach pain'
  elif s=='abdominal pain': return 'stomach pain'
  elif s=='gas': return 'bloating'
  elif s=='dry eyes': return 'dry eye'
  elif s=='chronic fatigue': return 'chronic pain'
  elif s=='muscle cramps': return 'muscle spasms'
  elif s=='right knee pain': return 'knee pain'
  elif s=='left knee pain': return 'knee pain'
  elif s=='palpatations': return 'palpitations'
  elif s=='heart palpitations': return 'palpitations'
  elif s=='heartburn': return 'acid reflux'
  elif s=='middle back pain': return 'back pain'
  elif s=='muscle twitching': return 'tremor'
  elif s=='shaking hands': return 'tremor'
  elif s=='low appetite': return 'loss of appetite'
  elif s=='decreased appetite': return 'loss of appetite'
  elif s=='low mood': return 'mental fatigue'
  elif s=='sleep disturbance': return 'sleep problems'
  elif s=='poor sleep': return 'unrefreshing sleep'
  elif s=='suicidal thoughts': return 'suicidal ideation'
  elif s=='memory loss': return 'memory problems'
  elif s=='pins and needles': return 'numbness'
  elif s=='numbness fingers': return 'numbness'
  elif s=='low motivation': return 'lack of motivation'
  elif s=='afternoon sleepiness': return 'excess daytime sleepiness'
  elif s=='mood': return 'anger'
  elif s=='sleep disturbance': return 'sleep problems'
  elif s=='cymbalta ' : return 'duloxetine'
  elif s=='levothyroxine' : return 'thyroxin'
  elif s=='omeprazole' : return 'Omeprazole'
  elif s=='plaquenil' : return 'hydroxychloroquine'
  elif s=='synthroid' : return 'thyroxin'
  elif s=='lyrica' : return 'pregabalin'
  elif s=='zofran' : return 'ondansetron'
  elif s=='wellbutrin' : return 'Bupropion'
  elif s=='celebrex' : return 'Celecoxib'
  elif s=='prozac' : return 'Fluoxetine'
  elif s=='aleve' : return 'Naproxen'
  elif s=='lexapro' : return 'Escitalopram'
  elif s=='mezavant' : return 'Mesalazine'
  elif s=='cbd' : return 'Cannabidiol'
  elif s=='adderall' : return 'Amphetamine'
  elif s=='azathioprine' : return 'Imuran'
  elif s=='xanax' : return 'Alprazolam'
  elif s=='seroquel' : return 'Quetiapine'
  elif s=='prilosec' : return 'Omeprazole'
  elif s=='wellbutrin xl' : return 'Bupropion'
  elif s=='lamictal' : return 'Lamotrigine'
  elif s=='klonopin' : return 'clonazepam'
  elif s=='cbd oil' : return 'Cannabidiol'
  elif s=='imuran' : return 'Imuran'
  elif s=='esomeprazole' : return 'Omeprazole'
  elif s=='dexamphetamine' : return 'Amphetamine'
  elif s=='cbd from hemp' : return 'Cannabidiol'
  elif s=='cbd capsules' : return 'Cannabidiol'
  elif s=='dexlansoprazole' : return 'lansoprazole'
  elif s=='cbd vape' : return 'Cannabidiol'
  elif s=='metoprolol tartrate' : return 'Metoprolol'
  elif s=='oxycodone lp' : return 'Oxycodone'
  elif s=='cbd cream' : return 'Cannabidiol'
  elif s=='klonopin prn' : return 'clonazepam'
  elif s=='quetiapine xr' : return 'Quetiapine'
  elif s=='cannabis cbd capsule 10mg' : return 'Cannabidiol'
  elif s=='puff w/ cbd' : return 'Cannabidiol'
  elif s=='lopressor' : return 'metoprolol'
  elif s=='dextroamphetamine' : return 'Amphetamine'
  elif s=='oxycodone-acetaminophen' : return 'Oxycodone'
  elif s=='omeprazole 20mg' : return 'Omeprazole'
  elif s=='prozac 20mg' : return 'Fluoxetine'
  elif s=='quetiapine modified release' : return 'Quetiapine'
  elif s=='amphetamine salts' : return 'Amphetamine'
  elif s=='prozac (morning)' : return 'Fluoxetine'
  elif s=='19:1 cbd:thc capsule' : return 'Cannabidiol'
  elif s=='klonopin odt' : return 'clonazepam'
  elif s=='cbd joint' : return 'Cannabidiol'
  elif s=='cannabis cbd oil' : return 'Cannabidiol'
  elif s=='supplement - cbd 2 x 2 + w/s pump' : return 'Cannabidiol'
  elif s=='19:1 thc:cbd vape pen' : return 'Cannabidiol'
  elif s=='cbd balm' : return 'Cannabidiol'
  elif s=='cbd:thc' : return 'Cannabidiol'
  elif s=='nano cbd' : return 'Cannabidiol'
  elif s=='lamictal 8am' : return 'Lamotrigine'
  elif s=='klonipin' : return 'clonazepam'
  elif s=='metoprolol2' : return 'Metoprolol'
  elif s=='methylphenidate er' : return 'methylphenidate'
  elif s=='ibs': return 'irritable bowel syndrome'
  elif s=='pots': return 'postural orthostatic tachycardia syndrome'
  elif s=='postural orthostatic tachycardia syndrome (pots)': return 'postural orthostatic tachycardia syndrome'
  elif s=='gerd': return 'gastroesophageal reflux disease'
  elif s=='pcos': return 'polycystic ovary syndrome'
  elif s=='ptsd': return 'post-traumatic stress disorder'
  elif s=='post-traumatic stress disorder (ptsd)': return 'post-traumatic stress disorder'
  elif s=='ocd': return 'obsessive compulsive disorder'
  else: return s
df['trackable_name'] = df['trackable_name'].apply(uni_name)
df['trackable_name'] = [i.lower() for i in df['trackable_name']]


# In[6]:


SymOrder = df[df['trackable_type']=='Symptom'].groupby('trackable_name').count().sort_values(by=['user_id'],ascending=False)
CondOrder = df[df['trackable_type']=='Condition'].groupby('trackable_name').count().sort_values(by=['user_id'],ascending=False)
TreatOrder = df[df['trackable_type']=='Treatment'].groupby('trackable_name').count().sort_values(by=['user_id'],ascending=False)


# In[7]:


def to_set(x):
    return set(x)

dfCondition = df[df['trackable_type']=='Condition'].groupby(['user_id','checkin_date','sex','age'])['trackable_name'].agg({'size': len, 'setCond': to_set})
dfSymptom = df[df['trackable_type']=='Symptom'].groupby(['user_id','checkin_date'])['trackable_name'].agg({'size': len, 'setSym': to_set})
dfTreatment = df[df['trackable_type']=='Treatment'].groupby(['user_id','checkin_date'])['trackable_name'].agg({'size': len, 'setTreat': to_set})

dfLabel  = pd.merge(dfCondition.reset_index(), dfSymptom.reset_index()[['user_id','checkin_date','setSym']], on=['user_id','checkin_date'])
dfLabel  = pd.merge(dfLabel, dfTreatment.reset_index()[['user_id','checkin_date','setTreat']], on=['user_id','checkin_date'])


# In[8]:


SexVariable = pd.get_dummies(dfLabel['sex']).values


# In[9]:


# non-correct list of input
listCond = list(CondOrder.index)
listSym = list(SymOrder.index)
listTreat = list(TreatOrder.index)

listCond = listCond[:150]
listSym = listSym[:150]
listTreat = listTreat[:100]


# In[10]:


from sklearn.externals import joblib
joblib.dump(CondOrder.index, 'CondOrder.pkl') 
joblib.dump(SymOrder.index, 'SymOrder.pkl')
joblib.dump(TreatOrder.index, 'TreatOrder.pkl')


# In[11]:


joblib.dump(listCond, 'listCond.pkl')
joblib.dump(listSym, 'listSym.pkl')
joblib.dump(listTreat, 'listTreat.pkl')


# In[12]:


Xage = np.array(list(dfLabel['age'])).reshape((-1,1))
Xsex = SexVariable
XCond = np.zeros((dfLabel.shape[0],len(listCond)))
XSym = np.zeros((dfLabel.shape[0],len(listSym)))
XTreat = np.zeros((dfLabel.shape[0],len(listTreat)))


# In[13]:


from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
le = preprocessing.LabelEncoder()
enc = OneHotEncoder(sparse=False)



# In[14]:


le.fit(listCond)
for i in range((dfLabel.shape[0])):
    listTEmpo = list(dfLabel.iloc[i]['setCond'])
    listTEmpo = [i for i in listTEmpo if i in listCond]
    if len(listTEmpo)!=0:
        indiceLoc = le.transform(listTEmpo)
        for j in indiceLoc:
            XCond[i,j]=1


# In[15]:


le.fit(listSym)
for i in range((dfLabel.shape[0])):
    listTEmpo = list(dfLabel.iloc[i]['setSym'])
    listTEmpo = [i for i in listTEmpo if i in listSym]
    if len(listTEmpo)!=0:
        indiceLoc = le.transform(listTEmpo)
        for j in indiceLoc:
            XSym[i,j]=1


# In[16]:


le.fit(listTreat)
for i in range((dfLabel.shape[0])):
    listTEmpo = list(dfLabel.iloc[i]['setTreat'])
    listTEmpo = [i for i in listTEmpo if i in listTreat]
    if len(listTEmpo)!=0:
        indiceLoc = le.transform(listTEmpo)
        for j in indiceLoc:
            XTreat[i,j]=1


# In[17]:


XFinal = np.concatenate((Xage, Xsex), axis=1)
XFinal = np.concatenate((XFinal, XCond), axis=1)
XFinal = np.concatenate((XFinal, XSym), axis=1)


# ## Splitting the Data
# We split the data into train (70%), test (15%), and validation (15%)

# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(XFinal, XTreat, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# ## Training the Model
# Model is a Random Forest Classifier
# We did a GridSearch to find the best hyperparameters which can be seen at the assignment of parameters:
# 
# parameters = {'criterion':('gini', 'entropy'), 'bootstrap':(True, False),'max_features':('auto','log2',20,40)}
# 
# As you can see below, we experimented with different models K-neighbors and Logistic Regression. 

# In[19]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


clf = RandomForestClassifier()
parameters = {'criterion':('gini', 'entropy'), 'bootstrap':(True, False),'max_features':('auto','log2',20,40)}

#clf = linear_model.LogisticRegression()
#parameters = {'C':(0.1,0.3,1,3,10)}

#clf  = neighbors.KNeighborsClassifier()
#parameters = {'n_neighbors':(2,3,4,5)}

#scoring = {'Accuracy': make_scorer(accuracy_score)}

grid = GridSearchCV(clf, parameters)

grid.fit(X_train, y_train)


# In[20]:


y_test_pred = grid.predict(X_test)


# # Accuracy
# We have an accuracy score of 77% on average with an F1 score of 0.91

# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[22]:


accuracy_score(y_test, y_test_pred)


# In[23]:


f1_score(y_test, y_test_pred, average='weighted')


# # Save our model

# In[24]:


from sklearn.externals import joblib
joblib.dump(grid, 'gridForestModel.pkl') 


# # Future Work and Experimentation Below

# In[25]:


#from sklearn.decomposition import PCA
#pca = PCA(n_components=20)
#pca.fit(XFinal)


# In[26]:


def TransformForPred(x):
    age = x[0]/50.0
    sex = x[1].lower()
    if s=='female': return np.asarray([0,1,0,0])
    elif s=='other': return np.asarray([0,0,0,1])
    elif s=='male': return np.asarray([0,0,1,0])
    elif s=='doesnt_say': return np.asarray([1,0,0,0])
    CondOrderIndex = joblib.load('CondOrder.pkl') 
    SymOrderIndex = joblib.load('SymOrder.pkl') 
    TreatOrderIndex = jobli
    b.load('TreatOrder.pkl')
    listCondRead = x[2]
    listSymRead = x[3]
    XCond = np.zeros((1,len(listCond)))
    XSym = np.zeros((1,len(listSym)))
   
    le = joblib.load('leCond.pkl')
    listCond=joblib.load('listCond.pkl')
    
    listTEmpo = listCondRead 
    listTEmpo = [i for i in listTEmpo if i in listCond]
    #print(listTEmpo)
    if len(listTEmpo)!=0:
        #print(listTEmpo)
        #print(le.transform(listTEmpo))
        #print(i)
        indiceLoc = le.transform(listTEmpo)
        for j in indiceLoc:
            XCond[i,j]=1
            
    le = joblib.load('leSym.pkl')
    listSym=joblib.load('listCond.pkl')

    listTEmpo = listSymRead
    listTEmpo = [i for i in listTEmpo if i in listSym]
    #print(listTEmpo)
    if len(listTEmpo)!=0:
        #print(listTEmpo)
        #print(le.transform(listTEmpo))
        #print(i)
        indiceLoc = le.transform(listTEmpo)
        for j in indiceLoc:
            XSym[i,j]=1


# In[ ]:





# In[27]:


#tempo=df[df['trackable_type']=='Treatment']
listSick = ['depression']
listSick = ['trackable_name','depression','anxiety','fatigue','generalized anxiety disorder',
            'post-traumatic stress disorder (ptsd)','ptsd','social anxiety disorder',
            'bipolar disorder','bipolar type ii','major depressive disorder',
            'obsessive-compulsive disorder','borderline personality disorder','c-ptsd',
            'stress','dysthymia','panic disorder','ocd','cptsd','social anxiety','bipolar type 1',
            'dissociative disorders','anxiety disorder','avoidant personality disorder',
            'irritability','mental health','atypical depression','psychosis','depressed mood',
            'dependent personality disorder','adult add','seasonal affective disorder (sad)',
            'schizotypal personality disorder','post traumatic stress disorder','mania',
            'low energy','mood','panic','schizoid personality disorder','low mood',
            'schizophrenia','body dysmorphic disorder','dissociative identity disorder',
            'complex ptsd','suicide and suicidal thoughts','mood disorders','suicidal thoughts',
            'depersonalization-derealization disorder','psychopathy','depression/anxiety',
            'general anxiety disorder','self harm','trouble anxieux g’©n’©ralis’©',
            'depressed','clinical depression','dark period','mental illness','teen depression',
            'self-injury/cutting','cranky','overwhelmed','suicidal ideation','major depression',
            'obsessive compulsive disorder']
tempo2 = df[df['trackable_name'].isin(listSick) ]
tempo2 = tempo2[['user_id','checkin_date']].drop_duplicates(['user_id','checkin_date'])
t1=pd.merge(df, tempo2[['user_id','checkin_date']], on=['user_id','checkin_date'])
t1.to_csv('OnlyMental.csv')
t1[t1['trackable_type']=='Symptom'].groupby('trackable_name').count().sort_values(by=['user_id'],ascending=False).to_csv('Symptom.csv')
t1[t1['trackable_type']=='Condition'].groupby('trackable_name').count().sort_values(by=['user_id'],ascending=False).to_csv('Condition.csv')
t1[t1['trackable_type']=='Treatment'].groupby('trackable_name').count().sort_values(by=['user_id'],ascending=False).to_csv('Treatment.csv')

