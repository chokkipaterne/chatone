import streamlit as st
from dotenv import load_dotenv
import random
import os
from time import time as now
from htmlTemplates import css, css_chat, bot_template, user_template
from processDocument import store_document, get_list_sources
from askDocument import ask_question
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
from constants import ODS, DATASET_EXTENSIONS, MAX_UPLOAD_FILE, DEFAULT_SEPERATOR, llms, openai_index
import io

load_dotenv()
__version__ = "0.1"
app_name = "ChatOne"
app_short_description = "Chat with any type of data in One place"
app_full_description = "Question answering system utilizing LLMs (GPT4ALL, LLAMA, LLAMA 2, Hugging Face, GPT3, GPT4) that allows you to ask questions on various document formats such as csv, xls, txt, pdf, eml, pptx, website page, open data, and even YouTube video."
source_docs = os.environ.get('SOURCE_DOCS', 'source_docs')
source_dbs = os.environ.get('SOURCE_DBS', 'source_dbs')
models_path = os.environ.get('MODELS_PATH')



# BOILERPLATE
#p267034
#summarise this document
st.set_page_config(layout='centered', page_title=f'{app_name}', page_icon=':books:')
st.header(f':books: {app_name} ({app_short_description})')
ss = st.session_state
st.write(f'{css_chat}', unsafe_allow_html=True)
df = None

if 'all_documents' not in ss: ss['all_documents'] = ['All Documents']
if 'filename_done' not in ss: ss['filename_done'] = None
if 'chat_history' not in ss: ss['chat_history']  = []
if 'chat_history1' not in ss: ss['chat_history1']  = []
if 'df' not in ss: ss['df'] = None
if 'list_sources' not in ss: ss['list_sources']  = []
if 'llm' not in ss:	ss['llm'] = llms[openai_index]
if 'sources' not in ss:	ss['sources'] = ""


# COMPONENTS
def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')

def ui_faq():
	st.markdown("---")
	st.markdown(
			"""
	## How does ChatOne operate?
	When you upload a document, it gets segmented into smaller sections and is stored within a specialized database termed a vector index. This index facilitates semantic search and retrieval.

	When a query is posed, ChatOne scours through the document segments, identifying the most pertinent ones using the vector index. Subsequently, it employs tools like GPT-3 to generate a conclusive response.

	## Why does document indexing take a significant amount of time?
	The duration required for document indexing can be extended if you're using a free OpenAI API key. This is due to the stringent rate limits placed on free API keys. To expedite the indexing process, utilizing a paid API key is recommended.

	## Are the provided answers completely accurate?
	No, the responses are not guaranteed to be 100% accurate. ChatOne utilizes resources like GPT-3 to craft answers. While GPT-3 is a robust language model, it can still exhibit errors and tendencies toward producing imaginative content (hallucinations). 

	However, for the majority of scenarios, ChatOne boasts a high level of accuracy and can address a wide array of questions. It's advisable to cross-reference answers with authoritative sources to confirm their correctness.

	"""
		)
	
def ui_about():
	#with st.expander('About'):
	st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) 🔑\n"  # noqa: E501
            "2. Upload your data on various document formats such as csv, xls, txt, pdf, eml, pptx, website page, and even YouTube video 📄\n"
            "3. Select the data on which you want to chat with💬\n"
            "4. Ask a question about your data💬\n"
        )
	st.markdown("---")
	st.markdown(f"""
	## About
	{app_full_description}
	""")
	st.markdown('Made by <a href="mailto:cpeterabiola@gmail.com">[Abiola Chokki]</a>', unsafe_allow_html=True)
	ui_faq()
	

def ui_settings():
	ui_model()
	ui_spacer(1)

def list_models():
	mdls = {}
	files = os.listdir(models_path)
	if len(files) > 0:
		for f in os.listdir(models_path):
			if os.path.isfile(models_path+f):
				mdls[f] = f
		print("list models")
		print(mdls)
		return mdls

def ui_model():
	models = []
	embeddings = []
	mdls = list_models()

	def format_func(option):
		return models[option]
	
	st.selectbox('LLM', llms, key='llm')
	st.text_input('OpenAI API key', value=os.environ.get("OPENAI_API_KEY"), type='password', key='api_key', disabled=(ss['llm'] != llms[openai_index]), on_change=on_api_key_change)

	if ss['llm'] == llms[0]:
		models = ["google/flan-t5-xxl", "databricks/dolly-v2-3b", "databricks/dolly-v2-7b", "Writer/camel-5b-hf", "Salesforce/xgen-7b-8k-base", "tiiuae/falcon-40b", "internlm/internlm-chat-7b"]
		embeddings = ['hkunlp/instructor-xl', 'hkunlp/instructor-large', 'sentence-transformers/all-MiniLM-L6-v2']
	elif ss['llm'] == llms[1]:
		models = mdls
		embeddings = ['sentence-transformers/all-MiniLM-L6-v2']
	elif ss['llm'] == llms[2]:
		models = mdls
		embeddings = ['sentence-transformers/all-MiniLM-L6-v2']
	elif ss['llm'] == llms[3]:
		models = mdls
		embeddings = ['sentence-transformers/all-MiniLM-L6-v2']
	elif ss['llm'] == llms[openai_index]:
		models = ['gpt-3.5-turbo','gpt-4','text-davinci-003','text-curie-001']
		#models = ['text-davinci-003','text-curie-001']
		embeddings = ['text-embedding-ada-002']
	st.selectbox('Embedding model', embeddings, key='embeddings_model_name')
	if ss['llm'] == llms[1] or ss['llm'] == llms[2] or ss['llm'] == llms[3]:
		st.selectbox('Main model', options=list(models.keys()), format_func= format_func, key='main_model', label_visibility="collapsed")
	else:
		st.selectbox('Main model', models, key='main_model')

def on_api_key_change():
	print(ss['api_key'])


def create_folder():
	directory_file = source_docs + "/" + ss.get('project_code')
	if not os.path.exists(directory_file):
		os.makedirs(directory_file)
	return directory_file + "/"

def load_txt_data(upload_file):
	directory_file = create_folder()
	data = upload_file.getvalue().decode()
	with open(directory_file + upload_file.name, 'w') as f:
		f.write(data)
	return data

def load_binary_data(upload_file):
	directory_file = create_folder()
	data = upload_file.getbuffer()
	with open(directory_file + upload_file.name, 'wb') as f:
		f.write(data)
	return data

def remove_html_tags(input):
    soup = BeautifulSoup(input, 'html.parser')
    return soup.get_text()

def remove_newlines(input):
    input = input.str.replace('\n', ' ')
    input = input.str.replace('\\n', ' ')
    input = input.str.replace('  ', ' ')
    input = input.str.replace('  ', ' ')
    return input

def load_html_data(url):
	directory_file = create_folder()
	fname = "html_" + ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)])+".txt"
	r = requests.get(url)
	with open(directory_file + fname, 'w') as file:
		data = remove_newlines(remove_html_tags(r.text).strip())
		file.write(data)
	return data

def load_csv_data(upload_file):
	directory_file = create_folder()
	df = pd.read_csv(upload_file)
	df.to_csv(directory_file + upload_file.name)
	return df

def on_upload_file_change():
	if ss['upload_file']:
		print(ss['upload_file'])
		file = ss['upload_file']
		
		if file.type == "text/csv":
			df = load_csv_data(file)
			ss['df'] = df
		else:
			try:
				load_txt_data(file)
			except:
				load_binary_data(file)

		ss['filename'] = ss['upload_file'].name
		if ss['filename'] != ss.get('filename_done'):
			with st.spinner(f'processing {ss["filename"]}...'):
				store_document(ss.get('project_code'), ss.get('llm'), ss.get('embeddings_model_name'), ss.get('main_model'), ss.get('api_key'))
				ss['filename_done'] = ss['filename']

def on_youtube_video_change():
	youtube_video_id = ss['youtube_video'].split('?v=')[-1]
	print(youtube_video_id)
	youtube_local_txt_file = loadTextFromYoutubeVideo(youtube_video_id)
	if youtube_local_txt_file != ss.get('filename_done'): 
		with st.spinner(f'processing {youtube_local_txt_file}...'):
			store_document(ss.get('project_code'), ss.get('llm'), ss.get('embeddings_model_name'), ss.get('main_model'), ss.get('api_key'))
			ss['filename_done'] = ss['filename']

def loadTextFromYoutubeVideo(youtube_video_id):
	transcript = YouTubeTranscriptApi.get_transcript(youtube_video_id)
	transcript_text = ""
	for entry in transcript:
		transcript_text += ' ' + entry['text']

	youtube_local_txt_file = youtube_video_id+"_youtube_transcript.txt"
	directory_file = create_folder()
	with open(directory_file + youtube_local_txt_file, 'w') as f:
		transcript_text = remove_newlines(transcript_text.strip())
		f.write(transcript_text)
	return youtube_local_txt_file

def generate_code():
	p_code = "p" + ''.join([str(random.randint(0, 999)).zfill(3) for _ in range(2)])
	return p_code

def on_project_code_change():
	print(ss.get('project_code'))
	#p678324
	ss['list_sources'] =  get_list_sources(ss.get('project_code'), ss.get('llm'), ss.get('embeddings_model_name'), ss.get('api_key'))
	#load filenames in select

def ui_project():
	st.write('Note down the following code in order to edit your project in the future.')
	if 'project_code' not in ss: 
		project_code = generate_code()
		ss['project_code'] = project_code
	st.text_input('Project Code', type='default', key='project_code', on_change=on_project_code_change, label_visibility="collapsed")

def on_page_url_change():
	print(ss['page_url'])

	url = ss['page_url']
	load_html_data(url)

	if ss['page_url'] != ss.get('filename_done'):
		with st.spinner(f'processing {ss["page_url"]}...'):
			store_document(ss.get('project_code'), ss.get('llm'), ss.get('embeddings_model_name'), ss.get('main_model'), ss.get('api_key'))
			ss['filename_done'] = ss['page_url']
			
def on_ods_data_change():
	print(ss['ods_data'])

def ui_document():
	st.markdown('Provide information about your data to be used')
	#st.file_uploader('Upload File', type=["csv", "doc", "docx", "enex", "eml", "epub", "md", "odt", "pdf", "ppt", "pptx", "txt"], key='upload_file', on_change=on_upload_file_change, label_visibility="collapsed")
	
	t1,t2,t3,t4 = st.tabs(['Upload File','YouTube Video', 'Webpage','OpenDataSoft'])
	with t1:
		st.file_uploader('Upload File', type=["csv", "doc", "docx", "enex", "eml", "epub", "md", "odt", "pdf", "ppt", "pptx", "txt"], key='upload_file', on_change=on_upload_file_change, label_visibility="collapsed")
	with t2:
		st.text_input('YouTube Video URL', placeholder="Enter URL of the YouTube video (e.g., https://www.youtube.com/watch?v=43MdhyYl2hY)", type='default', key='youtube_video', on_change=on_youtube_video_change, label_visibility="collapsed")
	with t3:
		st.text_input('Webpage', placeholder="Enter URL page (e.g., https://openai.com/research/overview)", type='default', key='page_url', on_change=on_page_url_change, label_visibility="collapsed")	
	with t4:
		st.text_input('Dataset Identifier', placeholder="Enter dataset identifier (e.g., tco-bus-circulation-passages-tr@keolis-rennes)", help="Go to https://data.opendatasoft.com/explore/dataset to select your dataset", type='default', key='dataset_id', on_change=on_ods_dataset_id_change, label_visibility="collapsed")	

def on_hide_source_change():
	print("source")

def ui_question():
	prompt_placeholder = st.container()
	with prompt_placeholder:
		cols = st.columns((6, 1))
		cols[0].text_area(
			'Question', key='question', height=100, placeholder='Type question here', help='', label_visibility="collapsed"
		)
		disabled = False if ss['question'] and len(ss.get('selected_docs'))>0 else True
		cols[1].button(
			"Submit", 
			type="primary", 
			on_click=on_click_callback
		)
		cols[1].checkbox('Hide sources', True, key='hide_source',help='show the sources used to answer question', on_change=on_hide_source_change)

def ui_select_docs():
	st.write('Select data to be used for asking questions')
	filenames = ['']
	sources = {}
	persist_directory = os.environ.get('SOURCE_DOCS', 'source_docs') + "/" +  ss['project_code'] + "/"
	for s in ss['list_sources']:
		sources[s]= s.replace(persist_directory,'')

	def on_change():
		ss['chat_history'] = []
		print(ss['selected_docs'])

	def format_func(option):
		return sources[option]

	st.multiselect('Select data', options=list(sources.keys()), format_func= format_func, on_change=on_change, key='selected_docs', label_visibility="collapsed")

def on_click_callback():
	with st.spinner("processing..."):
		answer, docs, ss['chat_history1'] = ask_question(ss['project_code'],ss['question'], ss.get('llm'), ss.get('embeddings_model_name'), ss.get('main_model'), ss.get('api_key'), ss.get('selected_docs'), False,False,ss['chat_history1'])
		ss['sources'] = ""
		if len(docs) > 0: ss['sources'] = show_sources(docs)
		message = answer
		ss['chat_history'].insert(0,message)
		message = ss['question']
		ss['chat_history'].insert(0,message)
		
def ui_bt_answer():
	c1,c2 = st.columns([2,1])
	disabled = False if ss['question'] and len(ss.get('selected_docs'))>0 else True

	#c2.checkbox('Hide sources', True, key='hide_source',help='show the sources used to answer question')

	#if c1.button('Process', disabled=disabled, type='primary', use_container_width=True):
		

def show_sources(docs):
	sources = ""
	for document in docs:
		sources += "<br/>-------------------------------------------------------"
		sources += "<br/>" + document.metadata["source"].split('/')[-1] + ":<br/>" + document.page_content
		sources += "<br/>-------------------------------------------------------<br/>"
	return sources

def show_chat_history():
	'''for i, message in enumerate(ss['chat_history']):
		if i % 2 == 0:
			st.write(user_template.replace(
				"{{MSG}}", message), unsafe_allow_html=True)
		else:
			st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)'''
	
	st.markdown("----")
	chat_placeholder = st.container()
	with chat_placeholder:
		for i, message in enumerate(ss['chat_history']):
			div = f"""
	<div class="chat-row 
		{'' if i % 2 == 1 else 'row-reverse'}">
		<img class="chat-icon" src="{'https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png' if i % 2 == 1 else 'https://www.pngall.com/wp-content/uploads/12/Avatar-Profile-PNG-Images.png'}"
			width=32 height=32>
		<div class="chat-bubble
		{'ai-bubble' if i % 2 == 1 else 'human-bubble'}">
			&#8203;{message}
		</div>
	</div>
			"""
			st.markdown(div, unsafe_allow_html=True)
		
		for _ in range(3):
			st.markdown("")

#Get delimiter of csv file
def detectDelimiter(header, can=False):
    if header.find(";") != -1:
        return ";"
    if header.find(",") != -1:
        return ","
    if can:
        return ""
    #default delimiter (MS Office export)
    return ";"

#Check if the link is downloadable
def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'zip' in content_type.lower():
        return {"success":False, "message":"The extension of the dataset is not supported"}
    elif 'html' in content_type.lower():
        return {"success":False, "message":"The extension of the dataset is not supported"}
    content_length = header.get('content-length', 0)
    content_length = int(content_length)
    if content_length and content_length > MAX_UPLOAD_FILE:  # 200 mb approx
        return {"success":False, "message":"The file size is greater than 200Mo"}
    return {"success":True, "message":""}

#convert xls, json, xlsx to csv with ;
def convert_file_to_csv(dataset_id, file_link, file_ext):
	file_ext = "."+file_ext
	rep_d = is_downloadable(file_link)
	if not rep_d["success"]:
		return rep_d
	r = requests.get(file_link, allow_redirects=True)
	content = r.content
	if not content or content is None:
		return None
		
	try:
		file_link = io.StringIO(content.decode('utf-8'))
		if (file_ext).lower() == ".csv":
			for header in file_link.getvalue().split('\n'):
				separator=detectDelimiter(header)
				break
			df = pd.read_csv(file_link, sep=separator, encoding='utf8')
		elif (file_ext).lower() == ".json":
			df = pd.read_json(file_link, encoding='utf8')
		elif (file_ext).lower() == ".xls" or (file_ext).lower() == ".xlsx":
			df = pd.read_excel(file_link, 0)
			
		directory_file = create_folder()
		save_path = directory_file+str(dataset_id)+".csv"
		df.to_csv(save_path,sep=DEFAULT_SEPERATOR, index=None)
		df = pd.read_csv(save_path)
		return df
	except Exception as e:
		print('Error details: '+ str(e))
		return None

#download data content
def on_ods_dataset_id_change():
	dataset_id = ss["dataset_id"]
	select_format = ""
	link = ODS["link"] + ODS["suffix"] + ODS["export_dt"]
	link = link.replace("#dt_id#", str(dataset_id))
	init_link = link
	rep_d = None

	for fmt in DATASET_EXTENSIONS:
		select_format = fmt
		link = init_link.replace("#format#", str(fmt))
		print(link)
		response = requests.get(link)
		if response and int(response.status_code) == 200:
			df = convert_file_to_csv(dataset_id, link, select_format)
			ss['df'] = df
			ss['filename'] = dataset_id
			if ss['filename'] != ss.get('filename_done'):
				with st.spinner(f'processing {ss["filename"]}...'):
					store_document(ss.get('project_code'), ss.get('llm'), ss.get('embeddings_model_name'), ss.get('main_model'), ss.get('api_key'))
					ss['filename_done'] = ss['filename']
				break
	return rep_d

# LAYOUT
col1, col2 = st.columns([6, 4])
with st.sidebar:
	ui_about()
	#ui_faq()

t1,t2, t3 = st.tabs(['💬 Chat with your data','📄 Upload your data', '🔑 Settings'])
with t3:
	ui_project()
	st.markdown("----")
	ui_settings()
with t2:
	ui_document()
	if ss['df'] is not None:
		st.dataframe(ss['df'])
with t1:
	ui_select_docs()
	ui_question()
	if not ss['hide_source'] and len(ss['sources']) > 0:
		with st.expander("Sources"):
			st.markdown(ss['sources'], unsafe_allow_html=True)
	show_chat_history()

	
	