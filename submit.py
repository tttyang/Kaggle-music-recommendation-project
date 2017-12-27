
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import math
import gc
print('Loading data...')
data_path = '/Users/yzh/Desktop/R data/kkbox/'
train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})
songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})
members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'},
                     parse_dates=['registration_init_time','expiration_date'])
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
print('Done loading...')


# In[32]:


gc.collect()


# In[4]:


print('Data merging...')


train = train.merge(songs, on='song_id', how='left')
test = test.merge(songs, on='song_id', how='left')

members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)

members['registration_year'] = members['registration_init_time'].dt.year
members['registration_month'] = members['registration_init_time'].dt.month
members['registration_date'] = members['registration_init_time'].dt.day

members['expiration_year'] = members['expiration_date'].dt.year
members['expiration_month'] = members['expiration_date'].dt.month
members['expiration_date'] = members['expiration_date'].dt.day
members = members.drop(['registration_init_time'], axis=1)

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.merge(songs_extra, on = 'song_id', how = 'left')
train.song_length.fillna(200000,inplace=True)
train.song_length = train.song_length.astype(np.uint32)
train.song_id = train.song_id.astype('category')


test = test.merge(songs_extra, on = 'song_id', how = 'left')
test.song_length.fillna(200000,inplace=True)
test.song_length = test.song_length.astype(np.uint32)
test.song_id = test.song_id.astype('category')

# import gc
# del members, songs; gc.collect();

print('Done merging...')


# In[5]:


print ("Adding new features")

def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids'].fillna('no_genre_id',inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)

def lyricist_count(x):
    if x == 'no_lyricist':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    return sum(map(x.count, ['|', '/', '\\', ';']))

train['lyricist'].fillna('no_lyricist',inplace=True)
test['lyricist'].fillna('no_lyricist',inplace=True)
train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)

def composer_count(x):
    if x == 'no_composer':
        return 0
    else:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1

train['composer'].fillna('no_composer',inplace=True)
test['composer'].fillna('no_composer',inplace=True)
train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)

def is_featured(x):
    if 'feat' in str(x) :
        return 1
    return 0

train['artist_name'].fillna('no_artist',inplace=True)
test['artist_name'].fillna('no_artist',inplace=True)
train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)

def artist_count(x):
    if x == 'no_artist':
        return 0
    else:
        return x.count('and') + x.count(',') + x.count('feat') + x.count('&')

train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)

# if artist is same as composer
train['artist_composer'] = (train['artist_name'] == train['composer']).astype(np.int8)
test['artist_composer'] = (test['artist_name'] == test['composer']).astype(np.int8)


# if artist, lyricist and composer are all three same
train['artist_composer_lyricist'] = ((train['artist_name'] == train['composer']) & (train['artist_name'] == train['lyricist']) & (train['composer'] == train['lyricist'])).astype(np.int8)
test['artist_composer_lyricist'] = ((test['artist_name'] == test['composer']) & (test['artist_name'] == test['lyricist']) & (test['composer'] == test['lyricist'])).astype(np.int8)

# is song language 17 or 45. 
def song_lang_boolean(x):
    if '17.0' in str(x) or '45.0' in str(x):
        return 1
    return 0

train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)


_mean_song_length = np.mean(train['song_length'])
def smaller_song(x):
    if x < _mean_song_length:
        return 1
    return 0

train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)
print("done")


# In[6]:


print("then adding magic play_ratio")
lentrain=len(train['song_id']);lentest=len(test['song_id'])

_dict_ratio_composer_played_train = {k: v for k, 
                v in ( train['composer'].value_counts()/len(train['composer']) ).iteritems() }
_dict_ratio_composer_played_test = {k: v for k, 
                v in ( test['composer'].value_counts()/len(test['composer']) ).iteritems() }
def ratio_composer_played_train(x): 
     if x=="no_composer" or x=="佚名":
        return 0.5/(lentrain)+0.5/lentest
     else:  
      try:
        return 0.5*_dict_ratio_composer_played_train[x]+0.5*_dict_ratio_composer_played_test[x]
      except KeyError:
        return _dict_ratio_composer_played_train[x]

def ratio_composer_played_test(x):
    if x=="no_composer" or x=="佚名":
         return 0.5/(lentrain)+0.5/lentest
    else:
      try:
        return 0.5*_dict_ratio_composer_played_train[x]+0.5*_dict_ratio_composer_played_test[x]
      except KeyError:
        return _dict_ratio_composer_played_test[x]

train['ratio_composer_played'] = train['composer'].apply(ratio_composer_played_train).astype(np.float64)
test['ratio_composer_played'] = test['composer'].apply(ratio_composer_played_test).astype(np.float64)
#######
_dict_ratio_artist_played_train = {k: v for k, 
                v in ( train['artist_name'].value_counts()/len(train['artist_name']) ).iteritems() }
_dict_ratio_artist_played_test = {k: v for k, 
                v in ( test['artist_name'].value_counts()/len(test['artist_name']) ).iteritems() }
def ratio_artist_played_train(x):
     if x=="no_artist" or x=="Various Artists" or x=="群星" or x=="佚名":
         return 0.5/(lentrain)+0.5/lentest
     else:
      try:
        return 0.5*_dict_ratio_artist_played_train[x]+0.5*_dict_ratio_artist_played_test[x]
      except KeyError:
        return _dict_ratio_artist_played_train[x]

def ratio_artist_played_test(x):
     if x=="no_artist" or x=="Various Artists" or x=="群星" or x=="佚名":
         return 0.5/(lentrain)+0.5/lentest
     else:
      try:
        return 0.5*_dict_ratio_artist_played_train[x]+0.5*_dict_ratio_artist_played_test[x]
      except KeyError:
        return _dict_ratio_artist_played_test[x]

train['ratio_artist_played'] = train['artist_name'].apply(ratio_artist_played_train).astype(np.float64)
test['ratio_artist_played'] = test['artist_name'].apply(ratio_artist_played_test).astype(np.float64)


# In[ ]:

_dict_ratio_song_played_train = {k: v for k, 
                v in ( train['song_id'].value_counts()/len(train['song_id']) ).iteritems() }
_dict_ratio_song_played_test = {k: v for k, 
                v in ( test['song_id'].value_counts()/len(test['song_id']) ).iteritems() }
def ratio_song_played_train(x):
  if x==np.nan:
      return 0.5/(lentrain)+0.5/lentest
  else:
    try:
      return 0.5*_dict_ratio_song_played_train[x]+             0.5*_dict_ratio_song_played_test[x]
    except KeyError:
      return _dict_ratio_song_played_train[x]

def ratio_song_played_test(x):
  if x==np.nan:
      return 0.5/(lentrain)+0.5/lentest
  else:
    try:
      return 0.5*_dict_ratio_song_played_train[x]             +0.5*_dict_ratio_song_played_test[x]
    except KeyError:
      return _dict_ratio_song_played_test[x]

train['ratio_song_played'] = train['song_id'].apply(ratio_song_played_train).astype(np.float64)
test['ratio_song_played'] = test['song_id'].apply(ratio_song_played_test).astype(np.float64)


# In[186]:

#songs['artist_name'].value_counts()/len(train['artist_name'])


print(" adding genre&lyricist ratio")
_dict_ratio_genre_played_train = {k: v for k, 
                v in ( train['genre_ids'].value_counts()/len(train['genre_ids']) ).iteritems() }
_dict_ratio_genre_played_test = {k: v for k, 
                v in ( test['genre_ids'].value_counts()/len(test['genre_ids']) ).iteritems() }
def ratio_genre_played_train(x):
    if x=="no_genre_id":
      return 0.5/(lentrain)+0.5/lentest
    else:
     try:
       return 0.5*_dict_ratio_genre_played_train[x]+0.5*_dict_ratio_genre_played_test[x]
     except KeyError:
       return _dict_ratio_genre_played_train[x]

def ratio_genre_played_test(x):
    if x=="no_genre_id":
      return 0.5/(lentrain)+0.5/lentest
    else:
      try:
        return 0.5*_dict_ratio_genre_played_train[x]+0.5*_dict_ratio_genre_played_test[x]
      except KeyError:
        return _dict_ratio_genre_played_test[x]

train['ratio_genre_played'] = train['genre_ids'].apply(ratio_genre_played_train).astype(np.float64)
test['ratio_genre_played'] = test['genre_ids'].apply(ratio_genre_played_test).astype(np.float64)

############
_dict_ratio_lyricist_played_train = {k: v for k, 
                v in ( train['lyricist'].value_counts()/len(train['lyricist']) ).iteritems() }
_dict_ratio_lyricist_played_test = {k: v for k, 
                v in ( test['lyricist'].value_counts()/len(test['lyricist']) ).iteritems() }
def ratio_lyricist_played_train(x):
     if x=="no_lyricist" or x=="佚名":
         return 0.5/(lentrain)+0.5/lentest
     else:
        try:
          return 0.5*_dict_ratio_lyricist_played_train[x]+0.5*_dict_ratio_lyricist_played_test[x]
        except KeyError:
          return _dict_ratio_lyricist_played_train[x]

def ratio_lyricist_played_test(x):
    if x=="no_lyricist" or x=="佚名":
         return 0.5/(lentrain)+0.5/lentest
    else:
      try:
        return 0.5*_dict_ratio_lyricist_played_train[x]+0.5*_dict_ratio_lyricist_played_test[x]
      except KeyError:
        return _dict_ratio_lyricist_played_test[x]

train['ratio_lyricist_played'] = train['lyricist'].apply(ratio_lyricist_played_train).astype(np.float64)
test['ratio_lyricist_played'] = test['lyricist'].apply(ratio_lyricist_played_test).astype(np.float64)
print("done")


# In[7]:


a=pd.concat([train.drop(['target'],axis=1),test.drop(['id'],axis=1)]).groupby("msno",as_index=False).agg({"artist_name":{"uni_art":pd.Series.nunique, #a is for user's play info
                                                                           "user_play":"count"}})
a.columns=a.columns.droplevel(level=0)
a=a.rename(columns={"":"msno"})
train=train.merge(a, on="msno", how="left")
test=test.merge(a, on="msno", how="left")
train["artist_habit"]=train["uni_art"]/train["user_play"]
test["artist_habit"]=test["uni_art"]/test["user_play"]


# In[8]:


user_aritst_count=pd.concat([train.drop(['target'],axis=1),test.drop(['id'],axis=1)]).groupby(["msno",
        "artist_name"],as_index=False).agg({"song_id":{"user_artist_count":"count"}})
user_aritst_count.columns=user_aritst_count.columns.droplevel(level=0)
user_aritst_count.columns.values[0]="msno"   
user_aritst_count.columns.values[1]="artist_name"
train=train.merge(user_aritst_count, on=["msno","artist_name"], how="left")
test=test.merge(user_aritst_count, on=["msno", "artist_name"],how="left")
train["coolartist_like"]=train["user_artist_count"]/train["user_play"]
test["coolartist_like"]=test["user_artist_count"]/test["user_play"]


# In[9]:


gc.collect()


# In[10]:


user_language_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["msno",
                "language"],as_index=False).agg({"song_id":{"cooluser_language_count":"count"}})
user_language_count.columns=user_language_count.columns.droplevel(level=0)
user_language_count.columns.values[0]="msno"
user_language_count.columns.values[1]="language"
train=train.merge(user_language_count, on=["msno","language"], how="left")
test=test.merge(user_language_count, on=["msno", "language"],how="left")
train["coollangu_like"]=train["cooluser_language_count"]/train["user_play"]
test["coollangu_like"]=test["cooluser_language_count"]/test["user_play"]


# In[11]:


unique_genre=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby("msno",
            as_index=False).agg({"genre_ids":{"cooluni_genre":pd.Series.nunique}})
unique_genre.columns=unique_genre.columns.droplevel(level=0)
unique_genre=unique_genre.rename(columns={"":"msno"})
train=train.merge(unique_genre, on="msno", how="left")
test=test.merge(unique_genre, on="msno", how="left")
train["coolgenre_habit"]=train["cooluni_genre"]/train["user_play"]
test["coolgenre_habit"]=test["cooluni_genre"]/test["user_play"]


# In[12]:


user_genre_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["msno",
"genre_ids"],as_index=False).agg({"song_id":{"cooluser_genre_count":"count"}})
user_genre_count.columns=user_genre_count.columns.droplevel(level=0)
user_genre_count.columns.values[0]="msno"
user_genre_count.columns.values[1]="genre_ids"
train=train.merge(user_genre_count, on=["msno","genre_ids"], how="left")
test=test.merge(user_genre_count, on=["msno", "genre_ids"],how="left")
train["coolgenre_like"]=train["cooluser_genre_count"]/train["user_play"]
test["coolgenre_like"]=test["cooluser_genre_count"]/test["user_play"]


# In[13]:


user_lyri_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["msno",
"lyricist"],as_index=False).agg({"song_id":{"cooluser_lyri_count":"count"}})
user_lyri_count.columns=user_lyri_count.columns.droplevel(level=0)
user_lyri_count.columns.values[0]="msno"
user_lyri_count.columns.values[1]="lyricist"
train=train.merge(user_lyri_count, on=["msno","lyricist"], how="left")
test=test.merge(user_lyri_count, on=["msno", "lyricist"],how="left")
train["coollyri_like"]=train["cooluser_lyri_count"]/train["user_play"]
test["coollyri_like"]=test["cooluser_lyri_count"]/test["user_play"]


# In[14]:


user_compo_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["msno",
"composer"],as_index=False).agg({"song_id":{"cooluser_compo_count":"count"}})
user_compo_count.columns=user_compo_count.columns.droplevel(level=0)
user_compo_count.columns.values[0]="msno"
user_compo_count.columns.values[1]="composer"
train=train.merge(user_compo_count, on=["msno","composer"], how="left")
test=test.merge(user_compo_count, on=["msno", "composer"],how="left")
train["coolcompo_like"]=train["cooluser_compo_count"]/train["user_play"]
test["coolcompo_like"]=test["cooluser_compo_count"]/test["user_play"]


# In[15]:


user_leng_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["msno"],
    as_index=False).agg({"song_length":{"cooluser_length_mean":"mean"}})
user_leng_count.columns=user_leng_count.columns.droplevel(level=0)
user_leng_count.columns.values[0]="msno"
train=train.merge(user_leng_count, on="msno", how="left")
test=test.merge(user_leng_count, on="msno",how="left")
train["coolleng_like"]=train["song_length"]/train["cooluser_length_mean"]
test["coolleng_like"]=test["song_length"]/test["cooluser_length_mean"]


# In[16]:


user_year_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["msno"],
    as_index=False).agg({"song_year":{"cooluser_year_mean":pd.Series.mean}})
user_year_count.columns=user_year_count.columns.droplevel(level=0)
user_year_count.columns.values[0]="msno"
train=train.merge(user_year_count, on="msno", how="left")
test=test.merge(user_year_count, on="msno",how="left")
train["coolyear_like"]=train["song_year"]-train["cooluser_year_mean"]
test["coolyear_like"]=test["song_year"]-test["cooluser_year_mean"]


# In[17]:


def convert(x):
    if x==0:
        return np.nan
    else:
        return x
train["bd2"]=train["bd"].apply(convert).astype(np.float64)
test["bd2"]=test["bd"].apply(convert).astype(np.float64)
song_bd_mean=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["song_id"],
    as_index=False).agg({"bd2":{"coolsong_bd_mean":pd.Series.mean},
                         "msno":{"coolsong_play":"count"}})
song_bd_mean.columns=song_bd_mean.columns.droplevel(level=0)
song_bd_mean.columns.values[0]="song_id"
train=train.merge(song_bd_mean, on="song_id", how="left")
test=test.merge(song_bd_mean, on="song_id",how="left")
train["coolbd_like"]=train["bd2"]-train["coolsong_bd_mean"]
test["coolbd_like"]=test["bd2"]-test["coolsong_bd_mean"]


# In[18]:


song_gender_count=pd.concat([train.drop(['target'],axis=1),
                    test.drop(['id'],axis=1)]).groupby(["song_id",
"gender"],as_index=False).agg({"msno":{"coolsong_gender_count":"count"}})
song_gender_count.columns=song_gender_count.columns.droplevel(level=0)
song_gender_count.columns.values[0]="song_id"
song_gender_count.columns.values[1]="gender"
train=train.merge(song_gender_count, on=["song_id","gender"], how="left")
test=test.merge(song_gender_count, on=["song_id", "gender"],how="left")
train["coolgender_like"]=train["coolsong_gender_count"]/train["coolsong_play"]
test["coolgender_like"]=test["coolsong_gender_count"]/test["coolsong_play"]


# In[19]:


user_info=pd.read_csv('/Users/yzh/Downloads/user_logs_final.csv')


# In[7]:


print("do")


# In[20]:


train = train.merge(user_info, on='msno', how='left')
test = test.merge(user_info, on='msno', how='left')


# train['user_artist']=(train['msno'].astype(object)+train['artist_name'].astype(object)).astype('category')
# test['user_artist']=(test['msno'].astype(object)+test['artist_name'].astype(object)).astype('category')

# train['user_genre']=(train['msno'].astype(object)+train['genre_ids'].astype(object)).astype('category')
# test['user_genre']=(test['msno'].astype(object)+test['genre_ids'].astype(object)).astype('category')

# In[21]:


print ("Train test and validation sets")
for col in train.columns:
    if train[col].dtype == object:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')


X_train = train.drop([#'num_25','num_50','num_985',
                      #'num_100',
                      #'days_listened',
    #'target','ratio_artist_played','ratio_composer_played', 
    #'ratio_lyricist_played', 'ratio_genre_played'], 
    'target','bd'], axis=1)
y_train = train['target'].values


X_test = test.drop([#'num_25','num_50','num_985',
                    #  'num_100', 'id'
                    #  'days_listened',
   # 'ratio_artist_played','ratio_composer_played', 
   # 'ratio_lyricist_played', 'ratio_genre_played'
                    'id','bd'], axis=1)
ids = test['id'].values


# del train, test; gc.collect();

d_train_final = lgb.Dataset(X_train, y_train)
watchlist_final = lgb.Dataset(X_train, y_train)
print('Processed data...')


# In[28]:


gc.collect()


# In[29]:


params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'learning_rate': 0.3,
        'verbose': 0,
        'num_leaves': 250,
#        'bagging_fraction': 0.95,
#        'bagging_freq': 1,
#        'bagging_seed': 1,
#        'feature_fraction': 0.9,
#        'feature_fraction_seed': 1,
        'max_bin': 256,
#       'max_depth': 10,
        'num_rounds': 150,
        'metric' : 'auc'
    }

model_f2 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=10)


# In[3]:


train.dtypes


# In[ ]:


p_test_2 = model_f2.predict(X_test)


# In[26]:


subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test_2
subm.to_csv(data_path + 'submission_lgbm_dart16.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

print('Done!')


# In[27]:


print("dd")


# In[34]:


gain = model_f2.feature_importance('gain')
ft = pd.DataFrame({'feature':model_f2.feature_name(), 'split':model_f2.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft


# In[37]:


import gc
gc.collect()


# In[11]:


X_train.dtypes


# In[12]:


X_test.dtypes

