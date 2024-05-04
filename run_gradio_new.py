import os
import gradio as gr
import argparse
import langchain
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from model import ChatGLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiRetrievalQAChain
from langchain.chains import RetrievalQA, LLMChain
from BERT_classification import BERT_Classification


parser = argparse.ArgumentParser()

parser.add_argument("--QA_vectordb", help="Persistent directory for the QA vectorDB.", type=str, default="./vectorDB/QA")
parser.add_argument("--drug_dict_vectordb", help="Persistent directory for the drug_dict vectorDB.", type=str, default="./vectorDB/drug")

parser.add_argument("--embedder", help="Directory for the embedder model.", type=str, default="../autodl-tmp/model/bge-m3")
parser.add_argument("--llm", help="Directory of the language model.", type=str, default="../autodl-tmp/model/chatglm3-6b")
parser.add_argument("--comparison_mode", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")

parser.add_argument("--bert_recommand", action="store_true") # lch bert
parser.add_argument("--context_cls", action="store_true") # clx bert
parser.add_argument("--cls_bert_dir", default="/root/autodl-tmp/model/cls_bert")

args = parser.parse_args()


langchain.verbose = args.verbose

class Model_center():
    def __init__(self):
        self.llm = ChatGLM_LLM(model_path=args.llm)
        embedder = HuggingFaceEmbeddings(model_name=args.embedder)
        
        self.QA_vectordb = Chroma(
            persist_directory=args.QA_vectordb,  
            embedding_function=embedder
        )
        
        self.drug_dict_vectordb = Chroma(
            persist_directory=args.drug_dict_vectordb,
            embedding_function=embedder
        )
        
        self.template = """你是何苦药店的导购员，具备全科医师的医学知识水平.请使用以下内容来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。尽量使答案简明扼要。下面是参考信息: 
——————————————————
{QA_context}
——————————————————
库存信息:{drug_dict_context}
问题:{question}
你的回答:"""
        self.prompt = PromptTemplate(
            input_variables=["QA_context", "drug_dict_context", "question"], template=self.template
        )
        
        self.QA_retriever = self.QA_vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 3, "score_threshold": 0.5},
        )
        
        self.drug_dict_retriever = self.drug_dict_vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 1, "score_threshold": 0.5},

        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        self.ifBert = 0
        
        ############################ TODO
        if args.bert_recommand:
            self.ifBert += 1
            cls_bert_dir = os.path.join(args.cls_bert_dir, 'bert_chinese_mc_base')
            context_cls_dir = os.path.join(args.cls_bert_dir, 'MLP_recognization.pt')
            bert_recommand_dir = os.path.join(args.cls_bert_dir, 'MLP_classification.pt')
            id_map = os.path.join(args.cls_bert_dir, 'id_label_mapping.csv')
            self.bert = BERT_Classification(bert_model = cls_bert_dir, mlp_c=bert_recommand_dir,
                                            mlp_b=context_cls_dir, id_map=id_map,  strategy='pooling')
        if args.context_cls:
            self.ifBert += 2
            cls_bert_dir = os.path.join(args.cls_bert_dir, 'bert_chinese_mc_base')
            context_cls_dir = os.path.join(args.cls_bert_dir, 'MLP_recognization.pt')
            bert_recommand_dir = os.path.join(args.cls_bert_dir, 'MLP_classification.pt')
            id_map = os.path.join(args.cls_bert_dir, 'id_label_mapping.csv')
            self.bert = BERT_Classification(bert_model = cls_bert_dir, mlp_c=bert_recommand_dir,
                                            mlp_b=context_cls_dir, id_map=id_map,strategy='pooling')
                        
        
    def naive_answer(self, question):
        prompt = self.template.replace('{context}', '').replace('{question}', question)
        return self.llm.invoke(prompt)
        
    
    def question_handler(self, question: str, chat_history: list =[], debug=args.debug, comparison_mode=args.comparison_mode):
        
        if question == None or len(question) < 1:
            return "", chat_history, ""
        
        # Step 1: context classification TODO use BERT
        cls_res = ""
        if args.context_cls:
            prompt = f"""请判断下列问题是否和询问病症或者药品相关。你的回答只能是'True'或者'False'.
            问题:{question}
            你的回答:"""
            question_list = []
            question_list.append(question)
            self.bert.embedded(question_list)
            is_related = (self.bert.classify())
            if(len(is_related) == 0):
                cls_res += "无法对该问题进行分类"
            else:
                if str(is_related) == "[0]":
                    cls_res += "该问题不是医疗问题"
                else:
                    cls_res += "该问题是医疗问题"
            
            
        rec_res = ""
        # Step 2: Bert medicine recommandation TODO  
        if args.bert_recommand:
            question_list = []
            question_list.append(question)
            self.bert.embedded(question_list)
            is_related = (self.bert.classify())
            bert_recommand_results = self.bert.predict()
            if(len(bert_recommand_results) == 0):
                rec_res += "无法对该问题进行药品推荐"
            else:
                rec_res += "根据您的问题，我们将推荐以下药品："
                for tmp in bert_recommand_results:
                    rec_res += (tmp + " ")
                    
        
        # Step 3: RAG
        QA_ref= self.QA_retriever.get_relevant_documents(question)
        drug_ref = self.drug_dict_retriever.get_relevant_documents(question)
        
        QA_context = []
        for QA_doc in QA_ref:
            QA_context.append(QA_doc.page_content)
        if len(QA_context) == 0:
            QA_context = "无参考信息。"
        else:
            QA_context = "\n\n".join(QA_context)
            
        drug_dict_context = []
        for drug_doc in drug_ref:
            drug_dict_context.append(drug_doc.page_content)
        if len(drug_dict_context) == 0:
            drug_dict_context = "无相关库存信息"
        else:
            drug_dict_context = "\n\n".join(drug_dict_context)
            
        prompt = self.prompt.invoke({"QA_context":QA_context, "drug_dict_context":drug_dict_context, "question":question})
        chain_result = self.chain.invoke({"QA_context":QA_context, "drug_dict_context":drug_dict_context, "question":question})['text']
        
        
        if comparison_mode:
            result = f"RAG result:\n{chain_result}\n"
            result += f"Naive result:\n{self.naive_answer(question)}"
        else:
            result = chain_result
        chat_history.append(
            (question, result))
        
        if debug:
            print("Is related test:", question, is_related)
            print("QA_context", QA_context)
            print("drug_dict_context", drug_dict_context)
            print("Prompt to RAG:", prompt)
            print("chain_result", chain_result)
        
        return_values = ["", chat_history]
        if args.context_cls:
            return_values.append(str(cls_res))
        if args.bert_recommand:
            return_values.append(str(rec_res))
        return tuple(return_values)
            
        
# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>何苦药店</center></h1>
                <center>何苦药店</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")
            if model_center.ifBert == 1:
                msg2 = gr.Textbox(label="bert分类结果")
            if model_center.ifBert == 2:
                msg3 = gr.Textbox(label="bert推荐结果")
            if model_center.ifBert == 3:
                msg2 = gr.Textbox(label="bert分类结果")
                msg3 = gr.Textbox(label="bert推荐结果")
            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        if model_center.ifBert == 0:
            db_wo_his_btn.click(model_center.question_handler, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        if model_center.ifBert == 1:
            db_wo_his_btn.click(model_center.question_handler, inputs=[
                            msg, chatbot], outputs=[msg, chatbot, msg2])
        if model_center.ifBert == 2:
            db_wo_his_btn.click(model_center.question_handler, inputs=[
                            msg, chatbot], outputs=[msg, chatbot, msg3])
        if model_center.ifBert == 3:
            db_wo_his_btn.click(model_center.question_handler, inputs=[
                            msg, chatbot], outputs=[msg, chatbot, msg2, msg3])
            
        
        
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
demo.launch()