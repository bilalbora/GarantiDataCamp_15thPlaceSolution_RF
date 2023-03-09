# Importing Libraries and Options

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
import warnings
warnings.simplefilter(action="ignore")

root = '/kaggle/input/garanti-bbva-data-camp/'
target = 'moved_after_2019'
idx = 'user_id'

# Reading Files

df_train = pd.read_csv(os.path.join(root, 'train_users.csv'))
df_test = pd.read_csv(os.path.join(root, 'test_users.csv'))
df_subm = pd.read_csv(os.path.join(root, 'submission.csv'))
df_lang = pd.read_csv(os.path.join(root, 'languages.csv'))
df_edu = pd.read_csv(os.path.join(root, 'education.csv'))
df_skill = pd.read_csv(os.path.join(root, 'skills.csv'))
df_exp = pd.read_csv(os.path.join(root, 'work_experiences.csv'))

df_train = df_train.set_index(idx)
df_test = df_test.set_index(idx)
df_subm = df_subm.set_index(idx)

df_exp = df_exp[df_exp['start_year_month'] < 201901]


# Education Process

df_edu = df_edu[df_edu['school_name'].notnull() & df_edu['degree'].notnull()]

df_edu = df_edu.drop(['start_year_month', 'end_year_month'], axis=1)

# Loc Degree
df_edu.loc[df_edu['degree'].str.contains('Yüksek Lis|Master|MSc|MS|Business Admini|MBA|M. Sc.|M.S.|Msc|M.S|M.Sc', na=False),
        'degree'] = 'Master'

df_edu.loc[df_edu['degree'].str.contains('Lisans|LİSANS|License|3,|3/|3.|2.|4.|/4|2,|4.0|4,0Lİsans|Graduate|Licence|Licenti|Bachelor|BCs|BsC|Üniversi|Muhendi|Licance|bachelor|Licence|Lisanas|BS|Bsc|lisans|B.Sc.|B. Sc.|B.S.|BA|BSc.|Engineer|Mühendis|mühendis|Fakülte|fakülte|FAKÜLTE|Faculty|FACULTY|faculty', na=False),
        'degree'] = 'Bachelor'

df_edu.loc[df_edu['degree'].str.contains('High School|Lise|lise|High school|Highschool|high|College|Lİse|Fen L', na=False),
        'degree'] = 'High_School'

df_edu.loc[df_edu['degree'].str.contains('Doctor|Doktora|PhD|PHD|Ph.D.|Phd|Post. Doc.|Ph. D.|Post|PH.D.|PHd|Ph.D', na=False),
        'degree'] = 'PhD'


# Dropping 10k datas which not includes normal degree
edu_idx = df_edu['degree'].value_counts().head(4).index
df_edu = df_edu[df_edu['degree'].isin(edu_idx)]

df_edu = df_edu.drop_duplicates(['user_id', 'degree'])
df_edu = pd.pivot(df_edu, index='user_id', columns='degree', values='school_name')
df_edu.head()

cols = ['Bachelor', 'Master', 'PhD']
for col in cols:
    df_edu.loc[df_edu[col].str.contains('Fırat|Firat', na=False),
        col] = 'Firat University'
    df_edu.loc[df_edu[col].str.contains('Yıldız|Yildiz|YILDIZ|yildiz', na=False),
        col] = 'YTU'
    df_edu.loc[df_edu[col].str.contains('Kocaeli|KOCAEL|kocael', na=False),
        col] = 'Kocaeli University'
    df_edu.loc[df_edu[col].str.contains('İstanbul Teknik Üniversitesi|Istanbul Technical University|İTÜ|ITU|Istanbul Teknik Üniversitesi', na=False),
        col] = 'ITU'
    df_edu.loc[df_edu[col].str.contains('İstanbul Üniver|Istanbul Univer', na=False),
        col] = 'Istanbul University'
    df_edu.loc[df_edu[col].str.contains('Sabanc|SABANC', na=False),
        col] = 'Sabanci University'
    df_edu.loc[df_edu[col].str.contains('Sakarya', na=False),
        col] = 'Sakarya University'
    df_edu.loc[df_edu[col].str.contains('Marmara', na=False),
        col] = 'Marmara University'
    df_edu.loc[df_edu[col].str.contains('Hacettepe', na=False),
        col] = 'Hacettepe University'
    df_edu.loc[df_edu[col].str.contains('Boğaziçi|Bogaz', na=False),
        col] = 'Bogazici University'
    df_edu.loc[df_edu[col].str.contains('Bilkent', na=False),
        col] = 'Bilkent University'
    df_edu.loc[df_edu[col].str.contains('Anadolu', na=False),
        col] = 'Anadolu University'
    df_edu.loc[df_edu[col].str.contains('Dokuz', na=False),
        col] = 'Dokuz Eylul University'
    df_edu.loc[df_edu[col].str.contains('Bilgi|BILGI', na=False),
        col] = 'Bilgi University'
    df_edu.loc[df_edu[col].str.contains('Bahceseh|Bahçe', na=False),
        col] = 'Bahcesehir University'
    df_edu.loc[df_edu[col].str.contains('Gebze Te', na=False),
        col] = 'GTU'
    df_edu.loc[df_edu[col].str.contains('ODTÜ|ODTU|Orta Doğu|Middle East', na=False),
        col] = 'ODTU'
    df_edu.loc[df_edu[col].str.contains('Ege', na=False),
        col] = 'Ege University'
    df_edu.loc[df_edu[col].str.contains('Gazi|GAZI', na=False),
        col] = 'Gazi University'
    df_edu.loc[df_edu[col].str.contains('Karadeniz', na=False),
        col] = 'KATU'
    df_edu.loc[df_edu[col].str.contains('Trakya', na=False),
        col] = 'Trakya University'
    df_edu.loc[df_edu[col].str.contains('Selçuk', na=False),
        col] = 'Selcuk University'
    df_edu.loc[df_edu[col].str.contains('Erciyes', na=False),
        col] = 'Erciyes University'
    df_edu.loc[df_edu[col].str.contains('Yeditepe', na=False),
        col] = 'Yeditepe University'
    df_edu.loc[df_edu[col].str.contains('Koç', na=False),
        col] = 'Koç University'

topteduidx = df_edu[
    df_edu['Bachelor'].isin(['YTU', 'ITU', 'Marmara University', 'Bilkent University', 'Bogazici University',
                                'GTU', 'Gazi University', 'Sabanci University', 'ODTU', 'Bilgi University',
                                 'Koç University', 'Galatasaray Üniversitesi'])].index.to_list()
for i in topteduidx:
    df_edu.loc[(df_edu.index == i), 'toptierdegflag'] = 1

df_edu['toptierdegflag'].fillna(0, inplace=True)
df_edu['toptierdegflag'] = df_edu['toptierdegflag'].astype('int64')

# Language Process

nan_index = df_lang[df_lang['proficiency'].isnull()].index.to_list()
for i in nan_index:
    df_lang.loc[(df_lang.index == i) & (df_lang['language'] == 'Turkish'), 'proficiency'] = 'native_or_bilingual'

df_lang = df_lang[df_lang['language'].notnull() & df_lang['proficiency'].notnull()]

dil_ranking = {
    'elementary': 1,
    'limited_working': 2,
    'professional_working': 3,
    'full_professional': 4,
    'native_or_bilingual': 5
}

df_lang.loc[:, 'proficiency'] = df_lang.loc[:, 'proficiency'].map(dil_ranking)

# Clean Shit Syntax
df_lang.loc[df_lang['language'].str.contains('Engl|ing|İng|engl|ENG|İNG|ING|Ingi|Ing|Engish', na=False), 'language'] = 'English'
df_lang.loc[df_lang['language'].str.contains('Türkçe|Turkis|Turk|Türk|turk|türk|TURK|TÜRK|Tük|YÖKDİ|Laz|Turc', na=False), 'language'] = 'Turkish'
df_lang.loc[df_lang['language'].str.contains('Alman|German|ALMA|GERM|alma|germ|Gerrm|Deut', na=False), 'language'] = 'German'
df_lang.loc[df_lang['language'].str.contains('Fren|Frans|fren|Fran|FRANS|FREN', na=False), 'language'] = 'French'
df_lang.loc[df_lang['language'].str.contains('Span|İspa|İSPAN|SPAN|spani|ispan|ıspan|ISPAN|Ispan|Espa|espa|ESPA', na=False), 'language'] = 'Spanish'
df_lang.loc[df_lang['language'].str.contains('Rusç|Russi|russ|RUSS|rusça|RUSÇA', na=False), 'language'] = 'Russian'
df_lang.loc[df_lang['language'].str.contains('ARAB|ARAP|arab|arap|Arap|Arab', na=False), 'language'] = 'Arabic'
df_lang.loc[df_lang['language'].str.contains('Itali|İtalyan|ITALİ|İTALY|ıtal|ital|İtali|Italy', na=False), 'language'] = 'Italian'
df_lang.loc[df_lang['language'].str.contains('Japan|Japon|japon|japan|JAPAN|JAPON', na=False), 'language'] = 'Japanese'
df_lang.loc[df_lang['language'].str.contains('Kore|KORE|kore', na=False), 'language'] = 'Korean'
df_lang.loc[df_lang['language'].str.contains('Kürt|Kurd|kürt|kurd', na=False), 'language'] = 'Kurdish'
df_lang.loc[df_lang['language'].str.contains('Chinese|Çince|CHINE|CHİNE|ÇİNCE|çince|chinese', na=False), 'language'] = 'Chinese'
df_lang.loc[df_lang['language'].str.contains('Portug|Porteki|PORTEK|PORTUGA', na=False), 'language'] = 'Portuguese'
df_lang.loc[df_lang['language'].str.contains('Sign|İşaret|işaret|sign|İŞARET|IŞARE|İSARE', na=False), 'language'] = 'Sign_Lang'
df_lang.loc[df_lang['language'].str.contains('Farsi|Farsça|Fars|FARS|fars', na=False), 'language'] = 'Farsi'
df_lang.loc[df_lang['language'].str.contains('Ukra|UKRAY|ukray|ukra', na=False), 'language'] = 'Ukrainian'
df_lang.loc[df_lang['language'].str.contains('Dutch|Flemenk|flemenk|hollan|dutch|Feleme', na=False), 'language'] = 'Dutch'
df_lang.loc[df_lang['language'].str.contains('Kazak|Kazakça|KAZAK|kazak', na=False), 'language'] = 'Kazakh'
df_lang.loc[df_lang['language'].str.contains('Bulgar|bulgar|BULGAR', na=False), 'language'] = 'Bulgarian'

### Val counts < 40 = 'Other'
oth_list = df_lang[df_lang['language'].isin(df_lang['language'].value_counts()[df_lang['language'].value_counts()<100].index)].language.unique()
for i in oth_list:
    df_lang.loc[df_lang['language'] == i, 'language'] = 'Other'

df_lang = df_lang.drop_duplicates(['user_id', 'language'])
df_lang = pd.pivot(df_lang, index='user_id', columns='language', values='proficiency')
df_lang = df_lang.fillna(0).astype(int)

df_lang['prof_sum'] = df_lang.sum(axis=1)

df_lang.loc[(df_lang['prof_sum'] <= 5), 'lang_level'] = 'very_low'
df_lang.loc[(df_lang['prof_sum'] > 5) & (df_lang['prof_sum'] < 10) , 'lang_level'] = 'low'
df_lang.loc[(df_lang['prof_sum'] >= 10) & (df_lang['prof_sum'] < 15) , 'lang_level'] = 'mid'
df_lang.loc[(df_lang['prof_sum'] >= 15) , 'lang_level'] = 'high'

# Skill Process

df_skill.loc[df_skill['skill'].str.contains('Python|PYTHON|python|piton', na=False), 'skill'] = 'Python'
df_skill.loc[df_skill['skill'].str.contains('Software Develop|Yazılım Geliş', na=False), 'skill'] = 'Software_Development'
df_skill.loc[df_skill['skill'].str.contains('Microsoft SQL', na=False), 'skill'] = 'SQL'
df_skill.loc[df_skill['skill'].str.contains('ASP.NET', na=False), 'skill'] = 'ASP.NET'
df_skill.loc[df_skill['skill'].str.contains('OOP|Object|Nesne Yönelim', na=False), 'skill'] = 'OOP'
df_skill.loc[df_skill['skill'].str.contains('JavaScript|JAVASCR|javasc', na=False), 'skill'] = 'JavaScript'
df_skill.loc[df_skill['skill'].str.contains('İngilizce|English|İNGİLİZ|ingiliz|ENGL|engli', na=False), 'skill'] = 'English'


will_use = df_skill['skill'].value_counts().head(50).index
df_skill = df_skill[df_skill['skill'].isin(will_use)]
df_skill['experience'] = True

df_skill = df_skill.drop_duplicates(['user_id', 'skill'])
df_skill = pd.pivot(df_skill, index='user_id', columns='skill', values='experience')
df_skill = df_skill.fillna(0).astype(int)
df_skill['skill_count'] = df_skill.sum(axis=1)

# Experience Process

df_exp = df_exp.sort_values(by=['user_id', 'start_year_month']) # we want nth function to return the last companies by date
df_new = pd.DataFrame()

df_new['company(1th)'] = df_exp.groupby(idx)['company_id'].nth(-1).astype(str)
df_new['company(2th)'] = df_exp.groupby(idx)['company_id'].nth(-2).astype(str)
df_new['company(3th)'] = df_exp.groupby(idx)['company_id'].nth(-3).astype(str)

df_new['company_location(1th)'] = df_exp.groupby(idx)['location'].nth(-1).astype(str)
df_new['company_location(2th)'] = df_exp.groupby(idx)['location'].nth(-2).astype(str)
df_new['company_location(3th)'] = df_exp.groupby(idx)['location'].nth(-3).astype(str)

df_new['min_exp_time'] = df_exp.groupby(idx)['start_year_month'].min()
df_new['max_exp_time'] = df_exp.groupby(idx)['start_year_month'].max()

df_new['company_count_2018'] = df_exp[df_exp['start_year_month'].gt(201712)].groupby(idx).size()
df_new['company_count_2017'] = df_exp[df_exp['start_year_month'].gt(201612)].groupby(idx).size()
df_new['company_count_2016'] = df_exp[df_exp['start_year_month'].gt(201512)].groupby(idx).size()

df_new['job_change_count'] = df_exp.groupby('user_id').agg({'company_id': 'nunique'})

df_new['max_date'] = df_new['max_exp_time'].apply(lambda x: str(x)[0:4]) + '-' + df_new['max_exp_time'].apply(lambda x: str(x)[4:6])
df_new['min_date'] = df_new['min_exp_time'].apply(lambda x: str(x)[0:4]) + '-' + df_new['min_exp_time'].apply(lambda x: str(x)[4:6])

df_new['max_date'] = df_new['max_date'].astype('datetime64')
df_new['min_date'] = df_new['min_date'].astype('datetime64')

todaydate = pd.to_datetime('2018-12-30')
df_new['exp_days'] = df_new['max_date'] - df_new['min_date']
df_new.loc[df_new['exp_days'] == '0 days', 'exp_days'] = todaydate - df_new['max_date']
df_new['exp_days'] = df_new['exp_days'].astype('O')
df_new['exp_days'] = df_new['exp_days'].apply(lambda x: str(x).split(' ')[0])
df_new.drop(['max_date', 'min_date'], axis=1, inplace=True)

df_new['exp_days'] = df_new['exp_days'].astype('int64')

df_new['dayby_comp_ratio'] = df_new['exp_days'] / df_new['job_change_count']

df_exp = df_new

# Merge All Tables

df_train[df_edu.columns] = df_edu[df_edu.columns]
df_train[df_lang.columns] = df_lang[df_lang.columns]
df_train[df_skill.columns] = df_skill[df_skill.columns]
df_train[df_exp.columns] = df_exp[df_exp.columns]

df_test[df_edu.columns] = df_edu[df_edu.columns]
df_test[df_lang.columns] = df_lang[df_lang.columns]
df_test[df_skill.columns] = df_skill[df_skill.columns]
df_test[df_exp.columns] = df_exp[df_exp.columns]

# FEATURE ENGINEERING

#Location Column

df_train.loc[df_train['location'].str.contains('Istanbul|İstanbul', na=False), 'location'] = 'Istanbul, Turkey'
df_train.loc[df_train['location'].str.contains('Ankara|ANKARA', na=False), 'location'] = 'Ankara, Turkey'
df_train.loc[df_train['location'].str.contains('İzmir|Izmir', na=False), 'location'] = 'İzmir, Turkey'
df_train.loc[df_train['location'].str.contains('Antalya', na=False), 'location'] = 'Antalya, Turkey'
df_train.loc[df_train['location'].str.contains('Konya', na=False), 'location'] = 'Konya, Turkey'
df_train.loc[df_train['location'].str.contains('Kocaeli|İzmit', na=False), 'location'] = 'Kocaeli, Turkey'
df_train.loc[df_train['location'].str.contains('Sakarya', na=False), 'location'] = 'Sakarya, Turkey'
df_train.loc[df_train['location'].str.contains('Konya', na=False), 'location'] = 'Konya, Turkey'
df_train.loc[df_train['location'].str.contains('Sivas', na=False), 'location'] = 'Sivas, Turkey'
df_train.loc[df_train['location'].str.contains('Eskişehir', na=False), 'location'] = 'Eskişehir, Turkey'
df_train.loc[df_train['location'].str.contains('Bursa', na=False), 'location'] = 'Bursa, Turkey'
df_train.loc[df_train['location'].str.contains('Muğla', na=False), 'location'] = 'Muğla, Turkey'
df_train.loc[df_train['location'].str.contains('Konya', na=False), 'location'] = 'Konya, Turkey'
df_train.loc[df_train['location'].str.contains('Adana', na=False), 'location'] = 'Adana, Turkey'
df_train.loc[df_train['location'].str.contains('Tekirda', na=False), 'location'] = 'Tekirdağ, Turkey'
df_train.loc[df_train['location'].str.contains('Edirne', na=False), 'location'] = 'Edirne, Turkey'
df_train.loc[df_train['location'].str.contains('Balıkes', na=False), 'location'] = 'Balıkes, Turkey'
df_train.loc[df_train['location'].str.contains('Kayser', na=False), 'location'] = 'Kayseri, Turkey'
df_train.loc[df_train['location'].str.contains('Aksara', na=False), 'location'] = 'Aksaray, Turkey'
df_train.loc[df_train['location'].str.contains('Manisa', na=False), 'location'] = 'Manisa, Turkey'
df_train.loc[df_train['location'].str.contains('Hatay', na=False), 'location'] = 'Hatay, Turkey'

df_test.loc[df_test['location'].str.contains('Istanbul|İstanbul', na=False), 'location'] = 'Istanbul, Turkey'
df_test.loc[df_test['location'].str.contains('Ankara|ANKARA', na=False), 'location'] = 'Ankara, Turkey'
df_test.loc[df_test['location'].str.contains('İzmir|Izmir', na=False), 'location'] = 'İzmir, Turkey'
df_test.loc[df_test['location'].str.contains('Antalya', na=False), 'location'] = 'Antalya, Turkey'
df_test.loc[df_test['location'].str.contains('Konya', na=False), 'location'] = 'Konya, Turkey'
df_test.loc[df_test['location'].str.contains('Kocaeli|İzmit', na=False), 'location'] = 'Kocaeli, Turkey'
df_test.loc[df_test['location'].str.contains('Sakarya', na=False), 'location'] = 'Sakarya, Turkey'
df_test.loc[df_test['location'].str.contains('Konya', na=False), 'location'] = 'Konya, Turkey'
df_test.loc[df_test['location'].str.contains('Sivas', na=False), 'location'] = 'Sivas, Turkey'
df_test.loc[df_test['location'].str.contains('Eskişehir', na=False), 'location'] = 'Eskişehir, Turkey'
df_test.loc[df_test['location'].str.contains('Bursa', na=False), 'location'] = 'Bursa, Turkey'
df_test.loc[df_test['location'].str.contains('Muğla', na=False), 'location'] = 'Muğla, Turkey'
df_test.loc[df_test['location'].str.contains('Konya', na=False), 'location'] = 'Konya, Turkey'
df_test.loc[df_test['location'].str.contains('Adana', na=False), 'location'] = 'Adana, Turkey'
df_test.loc[df_test['location'].str.contains('Tekirda', na=False), 'location'] = 'Tekirdağ, Turkey'
df_test.loc[df_test['location'].str.contains('Edirne', na=False), 'location'] = 'Edirne, Turkey'
df_test.loc[df_test['location'].str.contains('Balıkes', na=False), 'location'] = 'Balıkes, Turkey'
df_test.loc[df_test['location'].str.contains('Kayser', na=False), 'location'] = 'Kayseri, Turkey'
df_test.loc[df_test['location'].str.contains('Aksara', na=False), 'location'] = 'Aksaray, Turkey'
df_test.loc[df_test['location'].str.contains('Manisa', na=False), 'location'] = 'Manisa, Turkey'
df_test.loc[df_test['location'].str.contains('Hatay', na=False), 'location'] = 'Hatay, Turkey'

# Date Variables

df_train['start_year'] = df_train['min_exp_time'].apply(lambda x: str(x)[0:4])
df_train['start_year'].replace({'nan': -1}, inplace=True)
df_train['yeardiff'] = 2019 - df_train['start_year'].astype('int64')
df_train['yearby_jobchange_count'] = df_train['yeardiff'] / df_train['job_change_count']
df_test['start_year'] = df_test['min_exp_time'].apply(lambda x: str(x)[0:4])
df_test['start_year'].replace({'nan': -1}, inplace=True)
df_test['yeardiff'] = 2019 - df_test['start_year'].astype('int64')
df_test['yearby_jobchange_count'] = df_test['yeardiff'] / df_test['job_change_count']

df_train['lastexp_year'] = df_train['max_exp_time'].apply(lambda x: str(x)[0:4])
df_train['lastexp_month'] = df_train['max_exp_time'].apply(lambda x: str(x)[4:6])
df_train['start_month'] = df_train['min_exp_time'].apply(lambda x: str(x)[4:6])
df_test['lastexp_year'] = df_test['max_exp_time'].apply(lambda x: str(x)[0:4])
df_test['lastexp_month'] = df_test['max_exp_time'].apply(lambda x: str(x)[4:6])
df_test['start_month'] = df_test['min_exp_time'].apply(lambda x: str(x)[4:6])

df_train.drop(['start_month', 'lastexp_month'], axis=1, inplace=True)
df_test.drop(['start_month', 'lastexp_month'], axis=1, inplace=True)

df_train['daysperyear'] = df_train['exp_days'] / df_train['yeardiff']
df_test['daysperyear'] = df_test['exp_days'] / df_test['yeardiff']

df_train['exp_per_jcc'] = df_train['exp_days'] / df_train['yearby_jobchange_count']
df_test['exp_per_jcc'] = df_test['exp_days'] / df_test['yearby_jobchange_count']
df_train['exp_mul_jcc'] = df_train['exp_days'] * df_train['yearby_jobchange_count']
df_test['exp_mul_jcc'] = df_test['exp_days'] * df_test['yearby_jobchange_count']

# Create location flag columns and dropping high correlated cols
df_train.loc[df_train['location'].str.contains('Turkey', na=False), 'locFlag'] = 1
df_train['locFlag'].fillna(0, inplace=True)
df_train['locFlag'] = df_train['locFlag'].astype('int64')

df_test.loc[df_test['location'].str.contains('Turkey', na=False), 'locFlag'] = 1
df_test['locFlag'].fillna(0, inplace=True)
df_test['locFlag'] = df_test['locFlag'].astype('int64')

df_train.drop(['start_year', 'yeardiff', 'lastexp_year'], axis=1, inplace=True)
df_test.drop(['start_year', 'yeardiff', 'lastexp_year'], axis=1, inplace=True)

# Create loyalty rate with dayby_comp_ratio
df_train['loyalty_rate'] = pd.cut(df_train['dayby_comp_ratio'],
                                 bins=[10,264,517,968,18630],
                                 labels=[0,1,2,3])
df_test['loyalty_rate'] = pd.cut(df_test['dayby_comp_ratio'],
                                 bins=[10,264,517,968,18630],
                                 labels=[0,1,2,3])
df_train['loyalty_rate'] = df_train['loyalty_rate'].astype('O')
df_test['loyalty_rate'] = df_test['loyalty_rate'].astype('O')

# Create skill level with skill_count
df_train['skill_level'] = pd.cut(df_train['skill_count'],
                                 bins=[0,4,8,13,21,29],
                                 labels=[0,1,2,3,4])
df_test['skill_level'] = pd.cut(df_test['skill_count'],
                                 bins=[0,4,8,13,21,29],
                                 labels=[0,1,2,3,4])
df_train['skill_level'] = df_train['skill_level'].astype('O')
df_test['skill_level'] = df_test['skill_level'].astype('O')

# comp_home_sameloc variable was created to understand the effect of location

df_train.loc[df_train['company_location(1th)'].str.contains('Istanbul|İstanbul', na=False), 'company_location(1th)'] = 'Istanbul, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Ankara|ANKARA', na=False), 'company_location(1th)'] = 'Ankara, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('İzmir|Izmir', na=False), 'company_location(1th)'] = 'İzmir, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Antalya', na=False), 'company_location(1th)'] = 'Antalya, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Konya', na=False), 'company_location(1th)'] = 'Konya, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Kocaeli|İzmit', na=False), 'company_location(1th)'] = 'Kocaeli, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Sakarya', na=False), 'company_location(1th)'] = 'Sakarya, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Konya', na=False), 'company_location(1th)'] = 'Konya, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Sivas', na=False), 'company_location(1th)'] = 'Sivas, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Eskişehir', na=False), 'company_location(1th)'] = 'Eskişehir, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Bursa', na=False), 'company_location(1th)'] = 'Bursa, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Muğla', na=False), 'company_location(1th)'] = 'Muğla, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Konya', na=False), 'company_location(1th)'] = 'Konya, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Adana', na=False), 'company_location(1th)'] = 'Adana, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Tekirda', na=False), 'company_location(1th)'] = 'Tekirdağ, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Edirne', na=False), 'company_location(1th)'] = 'Edirne, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Balıkes', na=False), 'company_location(1th)'] = 'Balıkes, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Kayser', na=False), 'company_location(1th)'] = 'Kayseri, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Aksara', na=False), 'company_location(1th)'] = 'Aksaray, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Manisa', na=False), 'company_location(1th)'] = 'Manisa, Turkey'
df_train.loc[df_train['company_location(1th)'].str.contains('Hatay', na=False), 'company_location(1th)'] = 'Hatay, Turkey'

df_test.loc[df_test['company_location(1th)'].str.contains('Istanbul|İstanbul', na=False), 'company_location(1th)'] = 'Istanbul, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Ankara|ANKARA', na=False), 'company_location(1th)'] = 'Ankara, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('İzmir|Izmir', na=False), 'company_location(1th)'] = 'İzmir, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Antalya', na=False), 'company_location(1th)'] = 'Antalya, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Konya', na=False), 'company_location(1th)'] = 'Konya, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Kocaeli|İzmit', na=False), 'company_location(1th)'] = 'Kocaeli, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Sakarya', na=False), 'company_location(1th)'] = 'Sakarya, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Konya', na=False), 'company_location(1th)'] = 'Konya, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Sivas', na=False), 'company_location(1th)'] = 'Sivas, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Eskişehir', na=False), 'company_location(1th)'] = 'Eskişehir, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Bursa', na=False), 'company_location(1th)'] = 'Bursa, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Muğla', na=False), 'company_location(1th)'] = 'Muğla, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Konya', na=False), 'company_location(1th)'] = 'Konya, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Adana', na=False), 'company_location(1th)'] = 'Adana, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Tekirda', na=False), 'company_location(1th)'] = 'Tekirdağ, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Edirne', na=False), 'company_location(1th)'] = 'Edirne, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Balıkes', na=False), 'company_location(1th)'] = 'Balıkes, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Kayser', na=False), 'company_location(1th)'] = 'Kayseri, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Aksara', na=False), 'company_location(1th)'] = 'Aksaray, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Manisa', na=False), 'company_location(1th)'] = 'Manisa, Turkey'
df_test.loc[df_test['company_location(1th)'].str.contains('Hatay', na=False), 'company_location(1th)'] = 'Hatay, Turkey'

df_train['comp_home_sameloc'] = np.where((df_train['location'] == df_train['company_location(1th)']), 1,0)
df_test['comp_home_sameloc'] = np.where((df_test['location'] == df_test['company_location(1th)']), 1,0)

df_train['loc_loyalty'] = df_train.groupby(['comp_home_sameloc', 'loyalty_rate'])['dayby_comp_ratio'].transform('mean')
df_test['loc_loyalty'] = df_test.groupby(['comp_home_sameloc', 'loyalty_rate'])['dayby_comp_ratio'].transform('mean')

# Create yearly job_change_count on companies

df_train['2016_jcc_per'] = df_train['company_count_2016'] / df_train['job_change_count']
df_test['2016_jcc_per'] = df_test['company_count_2016'] / df_test['job_change_count']

df_train['2017_jcc_per'] = df_train['company_count_2017'] / df_train['job_change_count']
df_test['2017_jcc_per'] = df_test['company_count_2017'] / df_test['job_change_count']

df_train['2018_jcc_per'] = df_train['company_count_2018'] / df_train['job_change_count']
df_test['2018_jcc_per'] = df_test['company_count_2018'] / df_test['job_change_count']

# ENCODING

cat_cols = [col for col in df_test.columns if df_test[col].dtype == 'object']
num_cols = [col for col in df_test.columns if df_test[col].dtype != 'object']

for col in cat_cols:
    train_cats = set(df_train[col].unique())
    test_cats = set(df_test[col].unique())
    common_cats = set.intersection(train_cats, test_cats)

    df_train.loc[~df_train[col].isin(common_cats), col] = 'other'
    df_test.loc[~df_test[col].isin(common_cats), col] = 'other'

df_all = pd.concat([df_train, df_test], axis=0)

for col in cat_cols:
    df_all[col] = df_all[col].factorize()[0]

df_all[cat_cols] = df_all[cat_cols].astype('category')
df_all[num_cols] = df_all[num_cols].fillna(-1)

df_train = df_all.loc[df_train.index, df_train.columns]
df_test = df_all.loc[df_test.index, df_test.columns]

# TRAINING AND MODELING WITH BEST PARAMS

import time
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score ,StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
tqdm.pandas()

train = pd.read_csv('/kaggle/input/sets-final/train270_final.csv')
test = pd.read_csv('/kaggle/input/sets-final/test270_final.csv')

x = train.drop(['user_id','moved_after_2019'],axis=1)
y = train.moved_after_2019

#I created 5-fold cross-validation object to use for parameter tuning with Hyperopt, and 10-fold cross-validation object to use in general when performing cross-validation.

skf_hp = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=24)


def stratifiedKFold_on_acc(model, x, y, cv):
    start_time = time.time()
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring="accuracy", verbose=24, error_score='raise')
    print("Model : {}".format(model))
    print("scores", scores)
    print(f"scores mean: {np.mean(scores)} ,scores std :{np.std(scores)}")
    print("\n\n")
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken: {} seconds".format(total_time))


def stratifiedKFold_on_acc1(model, x, y, cv):
    start_time = time.time()
    scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring="accuracy", verbose=0, error_score='raise')
    print("Model : {}".format(model))
    print("scores", scores)
    print(f"scores mean: {np.mean(scores)} ,scores std :{np.std(scores)}")
    print("\n\n")
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken: {} seconds \n".format(total_time))
    return np.mean(scores)


def cv_seed_check(model, x, y, iteration, randomizer, splits):
    random_seeds = random.sample(range(randomizer), iteration)

    cv_means = []

    seeds = []

    for seed in tqdm(random_seeds):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

        mean_ = stratifiedKFold_on_acc1(model, x, y, skf)

        cv_means.append(mean_)

        seeds.append(seed)

        print(f'CV random_state::: {seed} :::\n\n')
        print('*-*-*-' * 20)

    cv_max_index = cv_means.index(max(cv_means))

    print(f'Max CV: {cv_means[cv_max_index]}, seed: {seeds[cv_max_index]}')
    print(f'Mean of all CV scores: {np.mean(cv_means)}, std: {np.std(cv_means)}\n')


def model_seed_check(x, y, iteration, randomizer, splits):
    random_seeds = random.sample(range(randomizer), iteration)

    cv_means = []

    seeds = []

    for seed in tqdm(random_seeds):
        model = RandomForestClassifier(n_estimators=200, random_state=seed)

        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=24)

        mean_ = stratifiedKFold_on_acc1(model, x, y, skf)

        cv_means.append(mean_)

        seeds.append(seed)

        print(f'Model random_state::: {seed} :::\n\n')
        print('*-*-*-' * 20)

    cv_max_index = cv_means.index(max(cv_means))

    print(f'Max CV: {cv_means[cv_max_index]}, seed: {seeds[cv_max_index]}')

model = RandomForestClassifier(n_estimators=200,random_state=24)

stratifiedKFold_on_acc(model,x,y,skf)

cv_seed_check(model,x,y,10,24,10)

# Hyperparameter Tuning

space = {
    'n_estimators': hp.choice('n_estimators', np.arange(50, 500, 50)),
    'max_depth': hp.choice('max_depth', [None] + list(np.arange(2, 20))),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'min_samples_split': hp.choice('min_samples_split', np.arange(2, 20)),
    'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 20)),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', np.arange(0.0, 0.5,0.1)),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', [None] + list(np.arange(5, 35, 5))),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', np.arange(0.0, 0.5)),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'class_weight': hp.choice('class_weight', [None, 'balanced', 'balanced_subsample']),
    'random_state': 24
}


def objective(params):
    model = RandomForestClassifier(**params)

    cv_results = cross_val_score(model.set_params(**params),
                                 x, y, cv=skf_hp ,scoring='accuracy',n_jobs=-1)

    best_acc = np.max(cv_results)
    return {'loss': -best_acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)

print("Best hyperparameters: ", best)

# CV Training and Stacking

accuracy = []

models = []

skf = StratifiedKFold(n_splits=21, shuffle=True, random_state=24)

cv_splits = list(skf.split(x, y))

for split_train, split_val in tqdm(cv_splits):
    split_train = x.index[split_train]
    split_val = x.index[split_val]

    x_train, y_train = x.loc[split_train], y.loc[split_train]
    x_val, y_val = x.loc[split_val], y.loc[split_val]

    print("Train shape:", x_train.shape, "|", "Val Shape:", x_val.shape)
    print("Positive Count in Val Split:", y_val.sum())

    model = RandomForestClassifier(**{'n_estimators': 200, 'random_state': 9})

    model.fit(x_train, y_train)

    preds = model.predict(x_val)

    acc = accuracy_score(y_val, preds)
    print("Fold Accuracy: ", acc)
    accuracy.append(acc)

    models.append(model)

    print("\n", "*-" * 50, "\n")

print(f'\n Mean Accuracy: {np.mean(accuracy)}, std: {np.std(accuracy)} \n')

# Predictions
model_preds = [model.predict(test) for model in models]

sub = pd.read_csv('/kaggle/input/garanti-bbva-data-camp/submission.csv')

sub.moved_after_2019 = (np.mean(model_preds,axis=0)>=0.5).astype(int)

sub.to_csv('sub_final_270feature_skf21_stack_rf_200_cv7900_pos3728.csv',index=False)

sub.moved_after_2019.value_counts()

# Feature Importances
importance = [model.feature_importances_ for model in models]


def plot_stack_feature_importance(importance, names, first_n=5):
    feature_importance = np.mean(importance, axis=0)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'].head(first_n), y=fi_df['feature_names'].head(first_n))
    plt.title(f'STACK FEATURE IMPORTANCE - FIRST {first_n} IMPORTANT FEATURES')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


plot_stack_feature_importance(importance, x.columns, 25)




























































































































