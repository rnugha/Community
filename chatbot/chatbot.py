import os
import json
import re, markdown
from typing import List, Dict, Any
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langdetect import detect
from deep_translator import GoogleTranslator

from .utils import get_vectorstore, init_env, ingest_pdfs_to_chroma
from .state import State
from .prompt import prompt_template
from .llm_init import call_groq_llama3

class Chatbot:
    def __init__(self):
        init_env()
        if not os.path.exists(os.path.join(os.getenv("CHROMA_PATH"), "index")):
            ingest_pdfs_to_chroma(os.getenv("DATA_PATH"), os.getenv("CHROMA_PATH"))

        self.vectorstore = get_vectorstore(os.getenv("CHROMA_PATH"))
        self.workflow = self._build_workflow()

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _summarize_query(self, question: str, chat_history: List[Dict]) -> str:
        if not chat_history:
            return question
            
        history_str = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in chat_history[-5:]
        )

        prompt = f"""Analyze this conversation and current question to create a standalone summarized query make it concise and information-rich:
        
        Conversation History:
        {history_str}
        
        Current Question: {question}
        
        Return only final query. 
        """
        
        try:
            response = call_groq_llama3([{"role": "system", "content": "You are a query optimizer"}, 
                                       {"role": "user", "content": prompt}])
            return response.strip('"\'') or question
        except Exception:
            return question
        
    def translate_to_ru(self, query: str) -> str:
        try:
            lang = detect(query)
            if lang in ["en", "kk"]:
                return GoogleTranslator(source='auto', target='ru').translate(query)
        except Exception:
            pass
        return query
    
    
    def _retrieve(self, state: State):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 10}
        )
        optimized_query = self._summarize_query(
            state["question"],
            state.get("chat_history", [])
        )
        translated_query = self.translate_to_ru(optimized_query)
        context = retriever.invoke(f"query: {translated_query}")
        return {"context": context}

        
    def _generate(self, state: State):
        context = self._format_docs(state["context"])
        question = state["question"]
        chat_history = state.get("chat_history", [])
        groq_messages = prompt_template(context, question, chat_history)

        try:
            response = call_groq_llama3(groq_messages)
            cleaned_response = self._format_answer_html(response)
            return {"answer": cleaned_response}
        except Exception as e:
            print(f"Generation error: {e}")
            return {"answer": "Sorry, an error occurred"}

    def _format_answer_html(self, response: str) -> str:
        try:
            response = json.loads(response)
        except Exception:
            response = response.strip()
            if (response.startswith('"') and response.endswith('"')) or (response.startswith("'") and response.endswith("'")):
                response = response[1:-1]
            response = response.replace('\\"', '"').replace("\\'", "'")

        response = response.replace("\n", "<br/>")

        response = re.sub(r'(https?://[^\s]+)', r'[\1](\1)', response)

        html = markdown.markdown(response, extensions=['tables'])
        return f'<div class="markdown-body">{html}</div>'


    def _build_workflow(self) -> CompiledStateGraph:
        graph = StateGraph(schema=State)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve")    
        graph.add_edge("retrieve", "generate")
        graph.set_finish_point("generate")
        return graph.compile()

    async def process_message(self, question: str, chat_history: list = None) -> str:
        payload = {
            "question": question,
            "chat_history": chat_history or [],
        }
        config = {"configurable": {"thread_id": 42}}

        try:
            events = []
            async for event in self.workflow.astream(payload, config=config):
                if isinstance(event, tuple):
                    _, event_data = event
                else:
                    event_data = event
                events.append(event_data)

            for event_data in events:
                if "generate" in event_data:
                    return event_data["generate"]["answer"]

            return "Sorry, I couldn't generate an answer."

        except Exception as e:
            return f"Error: {str(e)}"