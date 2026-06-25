import streamlit as st
import pickle
import re
import nltk
import gzip

nltk.download('punkt')

nltk.download('stopwords')

#loading models
with gzip.open("clf.pkl.gz", "rb") as f:
    clf = pickle.load(f)
tfidfd=pickle.load(open('tfidf.pkl','rb'))

def CleanResume(txt):
  cleanTxt=re.sub('http\S+\s',' ',txt)
  cleanTxt=re.sub('RT|cc',' ',cleanTxt)
  cleanTxt=re.sub('#\S+',' ',cleanTxt)
  cleanTxt=re.sub('@\S+',' ',cleanTxt)
  cleanTxt=re.sub('[%s]'%re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
  cleanTxt=re.sub('\s+',' ',cleanTxt)
  return cleanTxt


#web app
def main():
    st.title("Resume_Screening App")
    upload_file=st.file_uploader('Upload Resume',type=['txt','pdf'])
    
    if upload_file is not None:
        try:
            resume_bytes=upload_file.read()
            resume_txt=resume_bytes.decode('Utf-8')
        except UnicodeDecodeError:
            #if UTF-8 decoding fails,try decoding with 'latin-1'
            resuem_txt=resume_bytes.decode('latin-1')
        
        cleaned_resume=CleanResume(resume_txt)
        input_features=tfidfd.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]
        st.write(prediction_id)
        
        #map category name with category id
        category_mapping={
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            11: "Electrical Engineering",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name=category_mapping.get(prediction_id,'Unknown')
        st.write("Predicted Category:",category_name)
        print(category_name)