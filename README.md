# LangChain Simple RAG Pipeline:

This repo implements a simple RAG framework for a Pharmacy store chatbot. We use medical QA data from https://github.com/Toyhom/Chinese-medical-dialogue-data and a virtue inventory by web crawling from https://www.dayi.org.cn/ to build the knowledge database.

## Medical QA data filtering 


## Data processing and ChatGLM3 QA demo in `test.ipynb


## Step 0. Environment
Please first prepare an environment that can run ChatGLM3-6B.
Then, 
```bash

pip install langchain gradio chromadb sentence-transformers unstructured markdown
```

**If you want to launch the system with our provided models, for simplicity, please unzip the models under the relative path `../autodl-tmp/model/`**

## Step 1. Build VectorDB
```bash

cd RAG
conda activate nlp

```python

#### build VectorDB for demo QA files
python init_vectorDB.py --docs_dir ./docs/demo --persist_dir ./vectorDB/demo --embedder ../autodl-tmp/model/bge-m3

#### build VectorDB for complete QA files
python init_vectorDB.py --docs_dir ./docs/QA --persist_dir ./vectorDB/QA --embedder ../autodl-tmp/model/bge-m3 --chunk_size 100

#### build VectorDB for complete drug dictionary
python init_vectorDB.py --docs_dir ./docs/drug --persist_dir ./vectorDB/drug --embedder ../autodl-tmp/model/bge-m3 --chunk_size 100
```

We use bge-m3 as the embedder. Please specify the link to your embedder models by `--embedder`.

## Step 2. Launch Web-UI


# ##  chat with demo QA FILES & drug FILES
python run_gradio.py --comparison_mode --verbose --QA_vectordb ./vectorDB/demo --drug_dict_vectordb ./vectorDB/drug --embedder ../autodl-tmp/model/bge-m3 --llm ../autodl-tmp/model/chatglm3-6b

# ##  chat with complete QA & drug FILES
python run_gradio_new.py  --verbose --QA_vectordb ./vectorDB/QA --drug_dict_vectordb ./vectorDB/drug --embedder ../autodl-tmp/model/bge-m3 --llm ../autodl-tmp/model/chatglm3-6b 

# ##  chat with complete QA & drug FILES & Enable context classification and BERT recommmandation
python run_gradio_new.py  --verbose --QA_vectordb ./vectorDB/QA --drug_dict_vectordb ./vectorDB/drug --embedder ../autodl-tmp/model/bge-m3 --llm ../autodl-tmp/model/chatglm3-6b --bert_recommand --context_cls

```
 - `comparison_mode` will output the response with and without RAG.
 - `verbose` enables the verbose mode of LangChain. The full prompt will be displayed in the terminal.
 - `bert_recommand` will also load and output the recommendation using BERT in the web GUI.
 - `context_cls` will load and output the medical context classification using BERT in the web GUI.
 - `embedder` dir of the embedder models from Huggingface. Change it to the dir of your local embedder.
 - `llm` dir of the language model from Huggingface. Change it to the dir of your local llm.
 - `drug_dict_vectordb` the vector database for the inventory information.
 - ``

**Note that to display the BERT context classification and recommendation, please unzip our pretrained models under the relative path `../autodl-tmp/model/`**

【测试案例】

头痛吃什么药？

今天星期几？

夏天适合哪里旅游？

你们这里有感冒药吗？

维生素和抗生素还有库存吗？

钙片有没有卖？

这里有卖百多邦软膏吗？

我要买后悔药！

导购员，给我来一盒奥美拉唑！

高血压会遗传吗？

这几天发烧头痛、四肢无力怎么办？

我脾虚胃寒，湿气重，有什么药品推荐？
