# chatone
A Q&amp;A system leveraging LLMs (like GPT4ALL, LLAMA, GPT3, GPT4) for queries across diverse formats including csv, xls, pdf, web pages, and YouTube videos.
This is a proof of concept and might have some bugs, as not all features were fully tested due to constraints in testing various models.

## How to use
1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) 🔑
2. Upload your data on various document formats such as csv, xls, txt, pdf, eml, pptx, website page, and even YouTube video 📄
3. Select the data on which you want to chat with💬
4. Ask a question about your data💬

## How does ChatOne operate?
When you upload a document, it gets segmented into smaller sections and is stored within a specialized database termed a vector index. This index facilitates semantic search and retrieval.
When a query is posed, ChatOne scours through the document segments, identifying the most pertinent ones using the vector index. Subsequently, it employs tools like GPT-3 to generate a conclusive response.

## Why does document indexing take a significant amount of time?
The duration required for document indexing can be extended if you're using a free OpenAI API key. This is due to the stringent rate limits placed on free API keys. To expedite the indexing process, utilizing a paid API key is recommended.

## Are the provided answers completely accurate?
No, the responses are not guaranteed to be 100% accurate. ChatOne utilizes resources like GPT-3 to craft answers. While GPT-3 is a robust language model, it can still exhibit errors and tendencies toward producing imaginative content (hallucinations). 
However, for the majority of scenarios, ChatOne boasts a high level of accuracy and can address a wide array of questions. It's advisable to cross-reference answers with authoritative sources to confirm their correctness.

## Screenshots 
![Settings](/screenshots/settings.png)
Settings.

![Upload Data](/screenshots/data.png)
Upload Data.

![Chat](/screenshots/chat.png)
Chat With Data.