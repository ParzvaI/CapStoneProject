import streamlit as st
import pandas as pd
import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import itertools
from nltk.corpus import stopwords
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

def from_add(sample):
    sample=sample[sample.find("From: "):]
    return sample[:sample.find("\n")].split()[-1][1:][::-1][1:][::-1]

spam_add=['@cashette.com' ,'@gawab.com' ,'@mail.ru' ,'@freestuffo2.com' ,'@freestuffo1.com' ,'@freestuffo3.com' ,'@freestuffo4.com' 
    ,'@cash.com' ,'@tradedoubling.co.uk' ,'@users.1go.dk' ,'@one.lt' ,'@isuisse.com' ,'@iespana.es' ,'@mytrashmail.com' 
    ,'@bigmir.net' ,'@pornoroxx.net' ,'@freenet.de' ,'@mail15.com' ,'@fromru.com' ,'@hotpop.com' ,'@cashette.com' ,'@*.ru' 
    ,'@ukr.net' ,'@sibmail.com' ,'@thecannabishunter.com' ,'@advertfast.com' ,'@gawab.com' ,'@163.com' ,'@aichyna.com' 
    ,'@berahe.info' ,'@bi-dating.info' ,'@bk.ru' ,'@bookee.com' ,'@cashette.com' ,'@ccxt.info' ,'@chcb.info' ,'@corsa-tuning.' 
    ,'@deo-vindice' ,'@domain141.com' ,'@europe.com' ,'@fanaticars.info' ,'@faza.ru , @masterhost.ru' ,'@find-love.info' 
    ,'@for-fun.info' ,'@foteret.info' ,'@freefreemail.info' ,'@gawab.com' ,'@gold2world.biz' ,'@grifon.info' ,'@inbox.ru' 
    ,'@korsun.pp.ru' ,'@list.ru' ,'@mail.ru' ,'@mail333.com' ,'@moyareklama.ru' ,'@msk.su' ,'@muuh.info' ,'@myxost.com' 
    ,'@ne-quid-nimis.info' ,'@nil-admirari.info' ,'@octivian.com' ,'@pisem.net' ,'@pochta.ru' ,'@pooperduperz@gmail.com' 
    ,'@porn.com' ,'@portsaid.cc' ,'@prescrip.pl' ,'@punkass.com' ,'@qlfg.com' ,'@rambler.ru' ,'@sibmail.com' ,'@skim.com' 
    ,'@smeh.info' ,'@spambob.net' ,'@tele-vision.info' ,'@tut.by' ,'@ukr.net' ,'@vxaz.com' ,'@yandex.ru' ,'@yufz.com' 
    ,'@6url.com' ,'@bumpymail.com' ,'@cashette.com' ,'@centermail.com' ,'@centermail.net' ,'@discardmail.com' ,'@dodgeit.com' 
    ,'@e4ward.com' ,'@emailias.com' ,'@fakeinformation.com' ,'@front14.org' ,'@gawab.com' ,'@ghosttexter.de' ,'@gishpuppy.com' 
    ,'@greensloth.com' ,'@inbox.ru' ,'@jetable.org' ,'@kasmail.com' ,'@link2mail.net' ,'@mail.ru' ,'@mailexpire.com' 
    ,'@mailmoat.com' ,'@mailinator.com' ,'@mailnull.com' ,'@messagebeamer.de' ,'@mytrashmail.com' ,'@nervmich.net' 
    ,'@netmails.net' ,'@netzidiot.de' ,'@nurfuerspam.de' ,'@pookmail.com' ,'@portsaid.cc' ,'@privacy.net' ,'@punkass.com' 
    ,'@sneakemail.com' ,'@sofort-mail.de' ,'@sogetthis.com' ,'@spam.la' ,'@spambob.com' ,'@spambob.net' ,'@spambob.org' 
    ,'@spamday.com' ,'@spamex.com' ,'@spamgourmet.com' ,'@spamhole.com' ,'@spaminator.de' ,'@spammotel.com' ,'@spamtrail.com' 
    ,'@tempinbox.com' ,'@trash-mail.de' ,'@trashmail.net' ,'@ukr.net' ,'@xents.com' ,'@wuzup.net' ,'@zoemail.com' 
    ,'@marketingops.com' ,'@pisem.net' ,'@pleasantphoto.com' ,'@reitkopf.com' ,'@inbox.ru' ,'@mail333.com' ,'@mail.ru'
    ,'@list.ru' ,'@mail15.com' ,'@minelab.ru' ,'@fromru.com' ,'@xoxma.net' ,'@*.ru' ,'@mail.ru' ,'@cashette.com' 
    ,'@yandex.ru' ,'@kefir.000buy.com' ,'@bigfreemail.info' ,'@cashette.com' ,'@mail.ru' ,'@spambob' ,'@gawab.com' 
    ,'@bumpymail.com' ,'@centermail.com' ,'@centermail.net' ,'@discardmail.com' ,'@dodgeit.com' ,'@e4ward.com' 
    ,'@emailias.com' ,'@fakeinformation.com' ,'@front14.org' ,'@ghosttexter.de' ,'@jetable.net' ,'@kasmail.com' 
    ,'@link2mail.net' ,'@mailexpire.com' ,'@mailinator.com' ,'@mailmoat.com' ,'@messagebeamer.de' ,'@mytrashmail.com' 
    ,'@nervmich.net' ,'@netmails.net' ,'@netzidiot.de' ,'@nurfuerspam.de' ,'@privacy.net' ,'@punkass.com' ,'@sneakemail.com' 
    ,'@sofort-mail.de' ,'@sogetthis.com' ,'@spam.la' ,'@spamex.com' ,'@spamgourmet.com' ,'@spamhole.com' ,'@spaminator.de' 
    ,'@spammotel.com' ,'@spamtrail.com' ,'@trash-mail.de' ,'@trashmail.net' ,'@wuzup.net' ,'@portsaid.cc' ,'@sriaus.com' 
    ,'@ukr.net' ,'@pisem.net' ,'@mail333.com' ,'@gold-profits.info' ,'@sibmail.com' ,'@algerie.cc' ,'@blida.info' 
    ,'@mascara.ws' ,'@oran.cc' ,'@oued.info' ,'@oued.org' ,'@bahraini.cc' ,'@manama.cc' ,'@cameroon.cc' ,'@djibouti.cc' 
    ,'@timor.cc' ,'@alex4all.com' ,'@alexandria.cc' ,'@aswan.cc' ,'@banha.cc' ,'@giza.cc' ,'@ismailia.cc' ,'@mansoura.tv' 
    ,'@portsaid.cc' ,'@sharm.cc' ,'@sinai.cc' ,'@suez.cc' ,'@tanta.cc' ,'@zagazig.cc' ,'@eritrea.cc' ,'@guinea.cc' 
    ,'@najaf.cc' ,'@amman.cc' ,'@aqaba.cc' ,'@irbid.ws' ,'@jerash.cc' ,'@karak.cc' ,'@urdun.cc' ,'@zarqa.cc' ,'@kuwaiti.tv' 
    ,'@safat.biz' ,'@safat.info' ,'@safat.us' ,'@safat.ws' ,'@salmiya.biz' ,'@kyrgyzstan.cc' ,'@baalbeck.cc' ,'@hamra.cc' 
    ,'@lebanese.cc' ,'@lubnan.cc' ,'@lubnan.ws' ,'@agadir.cc' ,'@jadida.cc' ,'@jadida.org' ,'@maghreb.cc' ,'@marrakesh.cc' 
    ,'@meknes.cc' ,'@nador.cc' ,'@oujda.biz' ,'@oujda.cc' ,'@rabat.cc' ,'@tangiers.cc' ,'@tetouan.cc' ,'@dhofar.cc' 
    ,'@gabes.cc' ,'@ibra.cc' ,'@muscat.tv' ,'@muscat.ws' ,'@omani.ws' ,'@salalah.cc' ,'@seeb.cc' ,'@pakistani.ws' 
    ,'@falasteen.cc' ,'@hebron.tv' ,'@nablus.cc' ,'@quds.cc' ,'@rafah.cc' ,'@ramallah.cc' ,'@yunus.cc' ,'@abha.cc' 
    ,'@ahsa.ws' ,'@albaha.cc' ,'@alriyadh.cc' ,'@arar.ws' ,'@buraydah.cc' ,'@dhahran.cc' ,'@jizan.cc' ,'@jouf.cc' 
    ,'@khobar.cc' ,'@madinah.cc' ,'@qassem.cc' ,'@tabouk.cc' ,'@tayef.cc' ,'@yanbo.cc' ,'@dominican.cc' ,'@khartoum.cc' 
    ,'@omdurman.cc' ,'@sudanese.cc' ,'@hasakah.com' ,'@homs.cc' ,'@latakia.cc' ,'@palmyra.cc' ,'@palmyra.ws' ,'@siria.cc' 
    ,'@tajikistan.cc' ,'@bizerte.cc' ,'@gafsa.cc' ,'@kairouan.cc' ,'@nabeul.cc' ,'@nabeul.info' ,'@sfax.ws' ,'@sousse.cc' 
    ,'@tunisian.cc' ,'@ajman.cc' ,'@ajman.us' ,'@ajman.ws' ,'@fujairah.cc' ,'@fujairah.us' ,'@fujairah.ws' ,'@khaimah.cc' 
    ,'@sanaa.cc' ,'@yemeni.cc' ,'@zambia.cc' ,'@au.ru' ,'@bk.ru' ,'@fromru.ru' ,'@front.ru' ,'@go.ru' ,'@halyava.ru' 
    ,'@hotmail.ru' ,'@id.ru' ,'@inbox.ru' ,'@land.ru' ,'@list.ru' ,'@mailgate.ru' ,'@newmail.ru' ,'@nextmail.ru' ,'@nm.ru' 
    ,'@notmail.ru' ,'@ok.ru' ,'@pochta.ru' ,'@rambler.ru' ,'@ru.ru' ,'@sendmail.ru' ,'@yandex.ru' ,'@zmail.ru' 
    ,'@gomail.com.ua' ,'@mail15.com' ,'@algerie.cc' ,'@blida.info' ,'@mascara.ws' ,'@oran.cc' ,'@oued.info' ,'@oued.org' 
    ,'@bahraini.cc' ,'@manama.cc' ,'@cameroon.cc' ,'@djibouti.cc' ,'@timor.cc' ,'@alex4all.com' ,'@alexandria.cc' ,'@aswan.cc' 
    ,'@banha.cc' ,'@giza.cc' ,'@ismailia.cc' ,'@mansoura.tv' ,'@portsaid.cc' ,'@sharm.cc' ,'@sinai.cc' ,'@suez.cc' 
    ,'@tanta.cc' ,'@zagazig.cc' ,'@eritrea.cc' ,'@guinea.cc' ,'@najaf.cc' ,'@amman.cc' ,'@aqaba.cc' ,'@irbid.ws' ,'@jerash.cc' 
    ,'@karak.cc' ,'@urdun.cc' ,'@zarqa.cc' ,'@kuwaiti.tv' ,'@safat.biz' ,'@safat.info' ,'@safat.us' ,'@safat.ws' 
    ,'@salmiya.biz' ,'@kyrgyzstan.cc' ,'@baalbeck.cc' ,'@hamra.cc' ,'@lebanese.cc' ,'@lubnan.cc' ,'@lubnan.ws' ,'@agadir.cc' 
    ,'@jadida.cc' ,'@jadida.org' ,'@maghreb.cc' ,'@marrakesh.cc' ,'@meknes.cc' ,'@nador.cc' ,'@oujda.biz' ,'@oujda.cc' 
    ,'@rabat.cc' ,'@tangiers.cc' ,'@tetouan.cc' ,'@dhofar.cc' ,'@gabes.cc' ,'@ibra.cc' ,'@muscat.tv' ,'@muscat.ws' ,'@omani.ws' 
    ,'@salalah.cc' ,'@seeb.cc' ,'@pakistani.ws' ,'@falasteen.cc' ,'@hebron.tv' ,'@nablus.cc' ,'@quds.cc' ,'@rafah.cc' 
    ,'@ramallah.cc' ,'@yunus.cc' ,'@abha.cc' ,'@ahsa.ws' ,'@albaha.cc' ,'@alriyadh.cc' ,'@arar.ws' ,'@buraydah.cc' 
    ,'@dhahran.cc' ,'@jizan.cc' ,'@jouf.cc' ,'@khobar.cc' ,'@madinah.cc' ,'@qassem.cc' ,'@tabouk.cc' ,'@tayef.cc' 
    ,'@yanbo.cc' ,'@dominican.cc' ,'@khartoum.cc' ,'@omdurman.cc' ,'@sudanese.cc' ,'@hasakah.com' ,'@homs.cc' ,'@latakia.cc' 
    ,'@palmyra.cc' ,'@palmyra.ws' ,'@siria.cc' ,'@tajikistan.cc' ,'@bizerte.cc' ,'@gafsa.cc' ,'@kairouan.cc' ,'@nabeul.cc' 
    ,'@nabeul.info' ,'@sfax.ws' ,'@sousse.cc' ,'@tunisian.cc' ,'@ajman.cc' ,'@ajman.us' ,'@ajman.ws' ,'@fujairah.cc' 
    ,'@fujairah.us' ,'@fujairah.ws' ,'@khaimah.cc' ,'@sanaa.cc' ,'@yemeni.cc' ,'@zambia.cc' ,'@au.ru' ,'@bk.ru' ,'@fromru.ru' 
    ,'@front.ru' ,'@go.ru' ,'@halyava.ru' ,'@hotmail.ru' ,'@id.ru' ,'@inbox.ru' ,'@land.ru' ,'@list.ru' ,'@mailgate.ru' 
    ,'@newmail.ru' ,'@nextmail.ru' ,'@nm.ru' ,'@notmail.ru' ,'@ok.ru' ,'@pochta.ru' ,'@rambler.ru' ,'@ru.ru' ,'@sendmail.ru' 
    ,'@yandex.ru' ,'@zmail.ru' ,'@gomail.com.ua' ,'@mail15.com' ,'@bumpymail.com' ,'@centermail.com' ,'@centermail.net' 
    ,'@discardmail.com' ,'@dodgeit.com' ,'@e4ward.com' ,'@emailias.com' ,'@front14.org' ,'@ghosttexter.de' ,'@jetable.net' 
    ,'@jetable.org' ,'@kasmail.com' ,'@link2mail.net' ,'@mail333.com' ,'@mailblocks.com' ,'@maileater.com' ,'@mailexpire.com' 
    ,'@mailinator.com' ,'@mailmoat.com' ,'@mailnull.com' ,'@mailshell.com' ,'@mailzilla.com' ,'@messagebeamer.de' 
    ,'@mytrashmail.com' ,'@nervmich.net' ,'@netmails.net' ,'@netzidiot.de' ,'@nurfuerspam.de' ,'@pookmail.com' ,'@portsaid.cc' 
    ,'@privacy.net' ,'@punkass.com' ,'@shortmail.net' ,'@sibmail.com' ,'@sneakemail.com' ,'@sofort-mail.de' ,'@sogetthis.com' 
    ,'@spam.la' ,'@spambob.com' ,'@spambob.net' ,'@spambob.org' ,'@spamex.com' ,'@spamgourmet.com' ,'@spamhole.com' 
    ,'@spaminator.de' ,'@spammotel.com' ,'@spamtrail.com' ,'@tempinbox.com' ,'@trash-mail.de' ,'@trashmail.net' 
    ,'@bumpymail.com' ,'@centermail.com' ,'@centermail.net' ,'@discardmail.com' ,'@dodgeit.com' ,'@e4ward.com' 
    ,'@emailias.com' ,'@front14.org' ,'@ghosttexter.de' ,'@jetable.net' ,'@jetable.org' ,'@kasmail.com' ,'@link2mail.net' 
    ,'@mail333.com' ,'@mailblocks.com' ,'@maileater.com' ,'@mailexpire.com' ,'@mailinator.com' ,'@mailmoat.com' 
    ,'@mailnull.com' ,'@mailshell.com' ,'@mailzilla.com' ,'@messagebeamer.de' ,'@mytrashmail.com' ,'@nervmich.net' 
    ,'@netmails.net' ,'@netzidiot.de' ,'@nurfuerspam.de' ,'@pookmail.com' ,'@portsaid.cc' ,'@privacy.net' ,'@punkass.com' 
    ,'@shortmail.net' ,'@sibmail.com' ,'@sneakemail.com' ,'@sofort-mail.de' ,'@sogetthis.com' ,'@spam.la' ,'@spambob.com' 
    ,'@spambob.net' ,'@spambob.org' ,'@spamex.com' ,'@spamgourmet.com' ,'@spamhole.com' ,'@spaminator.de' ,'@spammotel.com' 
    ,'@spamtrail.com' ,'@tempinbox.com' ,'@trash-mail.de' ,'@trashmail.net']

disposable_add="  ".join(list(set(spam_add)))

def dispo(mail):
    if mail.split("@")[1] in disposable_add:
        return 1
    return 0

def body_extracter(smp):
    lz1=[]
    temp=[]
    for i in smp.split("\n"):
        temp.append(i)
        if ":" in i:
            if ord(i[0]) in range(65,91) and i[:i.find(":")].lower() not in ["http","https"] and " " not in i[:i.find(":")]:
                lz1.append(temp)
                temp=[]
    lz1.append(temp)
    lenz=[len(i) for i in lz1]
    stmp="\n".join(lz1[::-1][lenz[::-1].index(max(lenz))])
    if "Date" in stmp[:stmp.find(":")]:
        stmp="\n".join(map(str.lstrip,stmp.split("\n")[2:])) 
    return "\n".join([j for j in stmp[stmp.find(":")+1:].lstrip().split("\n") if len(j.lstrip())!=0])


def email_rem(s):
    ls=[]
    for i in s.split():
        if "@" not in i:
            ls.append(i)
    return " ".join(ls)

def lem(s):
    lemmatizer = WordNetLemmatizer() # Initializing Lemmatizer
    temp= []
    for w in s.split():
        temp.append(lemmatizer.lemmatize(w))
    return " ".join(temp)

def small(s):
    temp=[]
    for j in s.split():
        if len(j)>=3:                  
            temp.append(j)
    return " ".join(temp)


t5model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')



def summarize(text):
    preprocess_text = text.strip().replace("\n","")
    tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt").to(device) # return PyTorch tensors
    summary_ids = t5model.generate(tokenized_text,num_beams=4,no_repeat_ngram_size=2,min_length=15,max_length=25,early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)




def space(s):
    return " ".join([j.lstrip() for j in s.split("\n") if len(j.lstrip())!=0])
def cleaning(body,content):
    body=body.lower()
    content=content.lower()
    body= email_rem(body)
    content=(content)
    pun=['?', ':', '!', '$', '.', ',', '!', '|', '*', '&', '^', '%', '-', '#', '`', '~', '_', ';', '+', '+', '/', '<',
       '>', '{', '[', '}', ']',"<",'"','\\',")","(","\t"]
    pun1=['?', ':', '!', '$', '!', '|', '*', '&', '^', '%', '-', '#', '`', '~', '_', ';', '+', '+', '/', '<',")","("
       '>', '{', '[', '}', ']',"'s",'"','\\',")","(","\t"]
    for i,j in zip(pun1,pun):
        body=body.replace(i," ")
        content=content.replace(j," ")
    body=space(body)
    content=space(content)
    if len(body.lstrip())==0:
        body="None"
    content=small(content)
    content=lem(content)
    lem_body=lem(body)
    stop_words = list(stopwords.words('english'))
    for i in stop_words:
        regex = r"\b" + i + r"\b"
        content = content.replace(regex, '')
        body = body.replace(regex, '')
        lem_body=lem_body.replace(regex,'')
    t5model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    summa=summarize(body)
    return body,content,summa,lem_body
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
st.write("Email Intelligence")
ip=st.file_uploader(label="Upload File")
bt1=st.button("Load Data")
df3=pd.read_csv("cl.csv")
cv = CountVectorizer(max_features = 1000)
cv = cv.fit(df3['Content'])
X = cv.fit_transform(df3['Content'])
X=pd.DataFrame(X.toarray(),columns=cv.get_feature_names())
y = df3['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y)
clf = AdaBoostClassifier(n_estimators=100)
clf = clf.fit(X_train, y_train)
if bt1:
    email=open("C:/Users/Karthi/Desktop/Testing/"+ip.name,encoding="ISO-8859-1").read()
    body=body_extracter(email)
    from_a=from_add(email)
    dis=dispo(from_a)
    body=email_rem(body)
    b1,c1,s1,l1=cleaning(body,email)
    st.write("Details :   ")
    st.write("")
    st.write("")
    st.write("From      :    ",from_a)
    st.write("")
    st.write("")
    if clf.predict(cv.transform([l1]))==0:
        st.write("Category  :   Ham")
    else:
        st.write("Category  :   Spam")
    st.write("")
    st.write("")
    st.write("Summary   :   ",s1)