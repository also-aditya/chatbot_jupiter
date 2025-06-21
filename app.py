import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

# 1. Web Scraper
class JupiterFAQScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    @st.cache_data(show_spinner="Scraping Jupiter FAQ page...")
    def scrape_faq_page(_self, url: str) -> Optional[Dict[str, List[Dict]]]:
        """Scrapes FAQ data from Jupiter's help pages"""
        try:
            response = requests.get(url, headers=_self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            faq_content = soup.find('section', class_='prose')
            
            if not faq_content:
                st.error("FAQ content not found on the page")
                return None
            
            faq_data = {}
            current_section = None
            
            for element in faq_content.children:
                if element.name == 'h2':
                    current_section = element.get_text().strip()
                    faq_data[current_section] = []
                elif element.name in ['ol', 'ul']:
                    for li in element.find_all('li', recursive=False):
                        question, answer = _self._extract_qa_pair(li)
                        if question and current_section:
                            faq_data[current_section].append({
                                'question': question,
                                'answer': _self._clean_text(answer)
                            })
            
            return faq_data
        
        except Exception as e:
            st.error(f"Error scraping FAQ page: {e}")
            return None
    
    def _extract_qa_pair(self, li) -> tuple:
        """Extracts question and answer from list item"""
        question = None
        answer_parts = []
        
        strong_tags = li.find_all('strong')
        if strong_tags:
            question = strong_tags[0].get_text().strip()
            for content in strong_tags[0].next_siblings:
                if content.name == 'ul':
                    answer_parts.append(content.get_text().strip())
                elif content.string and content.string.strip():
                    answer_parts.append(content.string.strip())
        else:
            text_parts = li.get_text().strip().split('\n')
            if text_parts:
                question = text_parts[0]
                answer_parts = [part.strip() for part in text_parts[1:] if part.strip()]
        
        answer = ' '.join(answer_parts) if answer_parts else li.get_text().strip()
        return question, answer
    
    def _clean_text(self, text: str) -> str:
        """Cleans and normalizes text"""
        text = re.sub(r'\s+', ' ', text.replace('\n', ' '))
        return text.strip()

# 2. FAQ Indexing and Search
class JupiterFAQEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.faq_data = []
        self.section_map = []
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
    
    @st.cache_resource
    def load_model(_self):
        """Cache the sentence transformer model"""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def build_index(self, faq_data: Dict[str, List[Dict]]):
        """Process FAQ data and build FAISS index"""
        self.faq_data = []
        self.section_map = []
        questions = []
        
        # Flatten the FAQ data
        for section, qa_pairs in faq_data.items():
            for pair in qa_pairs:
                self.faq_data.append(pair)
                self.section_map.append(section)
                questions.append(self._preprocess_text(pair['question']))
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            self.model = self.load_model()
            embeddings = self.model.encode(questions, show_progress_bar=False)
        
        # Create FAISS index
        with st.spinner("Building search index..."):
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for better matching"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar questions"""
        query_embedding = self.model.encode([self._preprocess_text(query)])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i >= 0:  # FAISS may return -1 for invalid indices
                result = {
                    'question': self.faq_data[i]['question'],
                    'answer': self.faq_data[i]['answer'],
                    'section': self.section_map[i],
                    'confidence': float(1 / (1 + distance))  # Convert distance to confidence score
                }
                results.append(result)
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

# 3. Streamlit UI
class JupiterFAQBot:
    def __init__(self, faq_url: str):
        self.faq_url = faq_url
        self.scraper = JupiterFAQScraper()
        self.engine = JupiterFAQEngine()
        
        # Initialize Streamlit app
        st.set_page_config(
            page_title="Jupiter FAQ Bot",
            page_icon=":money_with_wings:",
            layout="wide"
        )
        
        self._initialize()
    
    def _initialize(self):
        """Load and index FAQ data"""
        st.title("Jupiter FAQ Assistant")
        st.markdown("Ask questions about Jupiter's AMC, AMB, and PRO services")
        
        with st.spinner("Loading FAQ data..."):
            faq_data = self.scraper.scrape_faq_page(self.faq_url)
            if faq_data:
                self.engine.build_index(faq_data)
                st.success(f"Loaded {len(self.engine.faq_data)} FAQs!")
            else:
                st.error("Failed to load FAQ data")
                st.stop()
    
    def render_sidebar(self):
        """Render sidebar with additional info"""
        with st.sidebar:
            st.header("About")
            st.markdown("""
                This bot answers questions about Jupiter's services using 
                their official FAQ content.
                
                It uses:
                - Web scraping to collect FAQs
                - NLP embeddings for understanding questions
                - Semantic search to find relevant answers
            """)
            
            st.markdown("---")
            st.markdown("**Popular questions:**")
            sample_questions = [
                "How can I avoid the AMB fee?",
                "What are the benefits of PRO account?",
                "When do the new fees take effect?",
                "What is the debit card annual fee?"
            ]
            
            for q in sample_questions:
                if st.button(q):
                    st.session_state.user_query = q
    
    def render_main(self):
        """Render main chat interface"""
        # Initialize session state
        if 'user_query' not in st.session_state:
            st.session_state.user_query = ""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input
        user_query = st.chat_input("Ask a question about Jupiter services")
        if user_query:
            st.session_state.user_query = user_query
        
        # Process query when available
        if st.session_state.user_query:
            self._process_query(st.session_state.user_query)
            st.session_state.user_query = ""  # Reset after processing
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])
    
    def _process_query(self, query: str):
        """Process user query and generate response"""
        # Add user query to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query
        })
        
        # Get bot response
        with st.spinner("Searching for answers..."):
            response = self._generate_response(query)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
    
    def _generate_response(self, query: str) -> str:
        """Generate response to user query"""
        if not query.strip():
            return "Please enter a question about Jupiter's services."
        
        results = self.engine.search(query)
        
        if not results or results[0]['confidence'] < 0.5:
            return self._get_fallback_response()
        
        return self._format_response(results)
    
    def _get_fallback_response(self) -> str:
        """Default response when no good match is found"""
        return """
        I couldn't find a specific answer to your question.  
        Here are some common topics you might ask about:
        - Average Monthly Balance (AMB) requirements
        - Debit Card Annual Fees  
        - PRO account benefits  
        
        Try asking about one of these or contact Jupiter support for more help.
        """
    
    def _format_response(self, results: List[Dict]) -> str:
        """Formats the search results into a response"""
        response = []
        
        # Main answer
        top_result = results[0]
        response.append(f"**{top_result['question']}**")
        response.append(f"{top_result['answer']}")
        response.append(f"*From section: {top_result['section']}*")
        
        # Related questions
        if len(results) > 1:
            response.append("\n**Related questions:**")
            for i, res in enumerate(results[1:3], 1):
                response.append(f"{i}. {res['question']}")
        
        return '\n\n'.join(response)
    
    def run(self):
        """Run the Streamlit application"""
        self.render_sidebar()
        self.render_main()

# Main execution
if __name__ == "__main__":
    FAQ_URL = "https://jupiter.money/amc-amb-pro-faqs/"
    
    try:
        bot = JupiterFAQBot(FAQ_URL)
        bot.run()
    except Exception as e:
        st.error(f"Failed to initialize FAQ bot: {e}")