import pandas as pd
import os
import string
import docx2txt
import re
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import PyPDF2

#import win32com.client

# # Convert DOC to DOCX format
# def covert_to_docx1(documents_path):
#     baseDir = documents_path
#     wrd = win32com.client.Dispatch("Word.Application")
#     for dir_path, dirs, files in os.walk(baseDir):
#         for file_name in files:
#             file_path = os.path.join(dir_path, file_name)
#             file_name, file_extension = os.path.splitext(file_path)
#             if file_extension.lower() == '.doc':
#                 docx_file = '{0}{1}'.format(file_path, 'x')
#                 if not os.path.isfile(docx_file):  # Skip conversion where docx file already exists
#                     print('Converting: {0}'.format(file_path))
#                     wordDoc = wrd.Documents.Open(file_path, False, False, False)
#                     wordDoc.SaveAs2(docx_file, FileFormat=16)
#                     wordDoc.Close()


# Complete cleaning pipeline
def cleaning_pipeline(documents_path):

    # setup data frame
    column_names = ('candidate_name', 'resume_body')
    df = pd.DataFrame(columns=column_names)

    # populate data frame
    populate_df(documents_path, df)

    # extract data
    populate_extracted_fields(df)
    df['cleaned'] = df[['resume_body', 'email_id', 'cities', 'companies', 'candidate_name']].apply(cleaner, axis=1)

    return df


# Populate the data frame
# Populates file name & raw resume body into the data frame from a given directory.
def populate_df(documents_path, dataframe):

    directory = os.path.normpath(documents_path)
    file_names = []
    resume_body = []

    count = 1

    for current_dir, sub_dirs_list, files in os.walk(directory):
        for file in files:
            if file.endswith(".docx"):
                file_names.append(file)
                file_txt = docx2txt.process(os.path.join(current_dir, file))
                resume_body.append(file_txt)
                print(str(count) + ' : ' + file)
                count += 1

            if file.endswith(".pdf"):
                file_names.append(file)
                file_txt = pdf_to_text(os.path.join(current_dir, file))
                resume_body.append(file_txt)
                print(str(count) + ' : ' + file)
                count += 1

    print('------------------------------------')
    print('number of files processed : ' + str(len(file_names)))

    dataframe['candidate_name'] = file_names
    dataframe['resume_body'] = resume_body
    dataframe['resume_body'] = dataframe['resume_body'].str.lower()


def pdf_to_text(pdf_file):

    # creating a pdf file object
    pdfFileObj = open(pdf_file, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    num_pages = pdfReader.numPages
    count = 0
    text = ""

    # The while loop will read each page
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count += 1
        text += pageObj.extractText()

    return text


# Extract and populate the DataFrame
def populate_extracted_fields(dataframe):
    dataframe['temp'] = dataframe['resume_body'].apply(extractor)
    dataframe['mobile_number'] = dataframe['temp'].apply(extract_mobile_number)
    dataframe['email_id'] = dataframe['temp'].apply(extract_email)
    dataframe['cities'] = dataframe['temp'].apply(extract_city)
    dataframe['skills'] = dataframe['temp'].apply(extract_skill)
    dataframe['companies'] = dataframe['temp'].apply(extract_company)
    dataframe.drop('temp', axis=1, inplace=True)


regex_mobile_number = re.compile('[0-9]{10}')
regex_email = re.compile('\w+\d*@\w+.com')

# Load the city list from CSV
city_list = []
with open('.\\CityList.csv') as f:
    city_list = f.readlines()
for i in range(len(city_list)):
    city_list[i] = city_list[i].rstrip('\n')

# Load the skill list from CSV
skill_list = []
with open('.\\SkillList.csv') as f:
    skill_list = f.readlines()
for i in range(len(skill_list)):
    skill_list[i] = skill_list[i].rstrip('\n')

# Load the company list from CSV
company_list = []
with open('.\\CompanyList.csv') as f:
    company_list = f.readlines()
for i in range(len(company_list)):
    company_list[i] = company_list[i].rstrip('\n')


# Extract from resume text
def extractor(text):

    # Extracts Mobile number, email , cities , skills & companies from raw text.
    text = text.lower()

    # Get mobile number list
    grab_mn_list = set(re.findall(regex_mobile_number, text))

    # Get email list
    grab_emails_list = set(re.findall(regex_email, text))

    grab_city_list = []  # mine this thing
    grab_skill_list = []  # mine this thing
    grab_company_list = []  # mine this thing

    grab = ["city", "skill", "company"]
    dict_preset = {"city": city_list, "skill": skill_list, "company": company_list}
    dict_grabbed = {"city": grab_city_list, "skill": grab_skill_list, "company": grab_company_list}

    # check for each city in the resume text
    for entity_label in grab:
        for item in dict_preset[entity_label]:
            regex_entity = re.compile(item)
            dict_grabbed[entity_label].extend(set(re.findall(regex_entity, text)))

    return (','.join(grab_mn_list),
            ','.join(grab_emails_list),
            ','.join(grab_city_list),
            ','.join(grab_skill_list),
            ','.join(grab_company_list))


# extract mobile number
def extract_mobile_number(ls):
    return ls[0]


# extract email
def extract_email(ls):
    return ls[1]


# extract city
def extract_city(ls):
    return ls[2]


# extract skill
def extract_skill(ls):
    return ls[3]


# extract company
def extract_company(ls):
    return ls[4]


def cleaner(cols):
    doc = cols[0]
    email_id = cols[1]
    cities = cols[2]
    companies = cols[3]
    cname = cols[4]

    # doc,email_id,cities,companies
    # Takes in a document as string of text, then performs the following:
    # 1. Remove all punctuation & numbers. Converts all words to lower case.
    # 2. Remove all stopwords
    # 3. Lemmatize the text py pos tagging.
    # 4. Returns the cleaned text

    stopwords_personal_info = ['nationality', 'sex', 'gender', 'marital', 'status', 'language', 'languages', 'personal',
                               'detail', 'date', 'birth', 'single', 'married', 'information', 'place', 'mother',
                               'tongue', 'known', 'know', 'male', 'female', 'email', 'contact', 'no.', 'number',
                               'residence', 'endorsements', 'comments', 'commented', 'liked', 'followers', 'follow',
                               'company', 'name']

    stopwords_languages = ['english', 'hindi', 'marathi', 'german', 'french', 'deutsch']

    stopwords_domain = ['roles', 'project', 'responsibilities', 'responsibility', "-", '-', '●', '–']

    stopwords_edu = ['x', 'xii', 'board', 'university', 'boarduniversity', 'high', 'higher', 'secondary', 'school',
                     'college', 'university', 'institute', 'technology', 'degree', 'year', 'passing', 'graduate',
                     'grade', 'academic', 'btech', 'bba']

    stopwords_declaration = ['declaration', 'hereby', 'declare', 'information', 'furnish', 'true', 'per', 'record',
                             'best']

    doc_1 = [word for word in doc.split()
             if word not in (' '.join(email_id))
             and word not in (' '.join(cities))
             and word not in (' '.join(companies))
             and word not in stopwords_personal_info
             and word not in stopwords_languages
             and word not in stopwords_domain and word not in stopwords_declaration
             and word not in stopwords_edu
             and word not in cname.lower()
             and not word.endswith(".com")]

    doc = ' '.join(doc_1)

    # list of characters
    num = '0 1 2 3 4 5 6 7 8 9'.split()
    loc = [char.lower() for char in doc if char not in string.punctuation and char not in num]

    # join them again for complete nopunc doc
    nopunc_doc = ''.join(loc)

    # remove stopwords
    stage2 = [word for word in nopunc_doc.split() if word not in stopwords.words('english')]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tagged = nltk.pos_tag(stage2)
    final = []

    for word, tag in tagged:
        if tag not in ['PRP', 'PRP$', 'CC', 'RB', 'RBR', 'RBS', 'TO', 'UH']:
            wn_tag = get_wordnet_pos(tag)
            if wn_tag is None:  # not supply tag in case of None
                lemma = lemmatizer.lemmatize(word)
            else:
                lemma = lemmatizer.lemmatize(word, pos=wn_tag)

            if lemma not in (' '.join(email_id)) and lemma not in (' '.join(cities)) and lemma not in (' '.join(
                    companies)) and lemma not in stopwords_personal_info and lemma not in stopwords_languages and lemma not in stopwords_domain and word not in stopwords_declaration and lemma not in stopwords_edu and lemma not in cname:
                final.append(lemma)

    return ' '.join(final)


# Clean resume body in DataFrame
# POS tagging helper function
# NLTK part of speech tagger uses treebank as default. To convert the tags to Wordnet lemmatizer acceptable tags,
# this function is required
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def extract_from_df(results_df):

    # SKILLS
    print('---------------------- SKILLS ------------------------')
    skills_dict = dict()
    skills_list = list(results_df['skills'])
    for i in range(0, len(skills_list)):
        val = skills_list[i]
        skills = val.split(',')
        for j in range(0, len(skills)):
            if skills[j] not in skills_dict:
                skills_dict[skills[j]] = 1
            else:
                skills_dict[skills[j]] += 1

    for key, value in skills_dict.items():
        print(str(key) + ' : ' + str(value))

    # CITIES
    print('---------------------- CITIES ------------------------')
    cities_dict = dict()
    cities_list = list(results_df['cities'])
    for i in range(0, len(cities_list)):
        val = cities_list[i]
        cities = val.split(',')
        for j in range(0, len(cities)):
            if cities[j] not in cities_dict:
                cities_dict[cities[j]] = 1
            else:
                cities_dict[cities[j]] += 1

    for key, value in cities_dict.items():
        print(str(key) + ' : ' + str(value))

    # COMPANIES
    print('--------------------- COMPANIES -------------------------')
    companies_dict = dict()
    companies_list = list(results_df['companies'])
    for i in range(0, len(companies_list)):
        val = companies_list[i]
        companies = val.split(',')
        for j in range(0, len(companies)):
            if companies[j] not in companies_dict:
                companies_dict[companies[j]] = 1
            else:
                companies_dict[companies[j]] += 1

    for key, value in companies_dict.items():
        print(str(key) + ' : ' + str(value))


def main():

    print('Start...')

    # Prepare the documents
    documents_path = 'D:\\Workarea\\AINADEL\\ResumeAnalytics\\training_data'
    #covert_to_docx1(documents_path)

    # Run pipeline
    df = cleaning_pipeline(documents_path)

    print('Processing...')

    # Extract data from dataframe
    extract_from_df(df)

    # Write results to CSV
    df.to_csv('D:\\Workarea\\AINADEL\\ResumeAnalytics\\results.csv')

    print('Processing Complete...')


main()
