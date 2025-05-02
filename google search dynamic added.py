import os
import json
import sqlite3
import sys
import logging
import uuid
import langid
from typing import Tuple
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2 import Memory
import datetime
from prompt_template import prompt_temp  # Import the prompt template
# Import translator functionality
from translator import LanguageTranslator, show_language_menu
from tools.hackernewsarticles.hackernew import get_top_hackernews_stories
from tools.SearchEngine.GotResults import main as search_engine_main

# Load environment variables from .env file
load_dotenv()

# Configure logging
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Load model configuration from environment variables
api_key = os.getenv('MODEL_API_KEY', 'sk-27185f30930a4613b26d90666082c23b')
base_url = os.getenv('MODEL_BASE_URL', 'http://80.188.223.202:10101/v1')
model_id = os.getenv('MODEL_ID', 'pawan941394/hind-ai:latest')

# Feature flags from environment variables
enable_reasoning = os.getenv('ENABLE_REASONING', 'true').lower() == 'true'
enable_translation = os.getenv('ENABLE_TRANSLATION', 'true').lower() == 'true'
show_tool_calls = os.getenv('SHOW_TOOL_CALLS', 'true').lower() == 'true'
enable_search = os.getenv('ENABLE_SEARCH', '').lower() == 'true'  # Default to empty, will ask user

# Memory settings
max_history_runs = int(os.getenv('MAX_HISTORY_RUNS', '15'))

# Import the reasoning module
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools', 'resioning'))
try:
    from tools.resioning.reasioning import get_reasoning_for_prompt
except ImportError:
    logger.warning("Reasoning module not found. Reasoning functionality will be disabled.")
    
    # Fallback function if reasoning module is not available
    def get_reasoning_for_prompt(query: str, chat_history: str = "") -> Tuple[str, str]:
        return "", ""

# Initialize translator
try:
    translator = LanguageTranslator()
except ImportError:
    logger.warning("Translator module not found. Translation functionality will be disabled.")
    
    # Define a placeholder translator class
    class DummyTranslator:
        def translate_text(self, text, source_lang, target_lang):
            return text
    
    translator = DummyTranslator()

# User language settings (load from environment if available)
user_language = {
    "code": os.getenv('DEFAULT_LANGUAGE_CODE', 'eng_Latn'),
    "name": os.getenv('DEFAULT_LANGUAGE_NAME', 'English')
}

def web_search(query: str) -> str:
    """Search the web for the given query and return summarized results.
    
    Args:
        query (str): The search query string.
        
    Returns:
        str: JSON string containing search results with titles, URLs, and summaries.
    """
    results = search_engine_main(query)
    # Format the results in a readable way for the agent
    formatted_results = []
    for i in range(len(results['titles'])):
        formatted_results.append({
            "title": results['titles'][i],
            "url": results['urls'][i],
            "summary": results['summary'][i]
        })
    return json.dumps(formatted_results, ensure_ascii=False)

# Helper function to sanitize filenames
def sanitize_filename(filename):
    """Sanitize a string to be used as a filename"""
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def create_user_database_dir(username):
    """Create directory structure for user database: Users/{username}/"""
    user_dir = os.path.join("Users", username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_sessions_for_user(username):
    """Get list of available sessions for the user from the database"""
    user_dir = os.path.join("Users", username)
    if not os.path.exists(user_dir):
        return []
    
    db_path = os.path.join(user_dir, f"{username}_chats.db")
    
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_sessions'")
        if not cursor.fetchone():
            conn.close()
            return []
            
        cursor.execute("SELECT DISTINCT session_id FROM agent_sessions")
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    except Exception as e:
        print(f"Error checking sessions: {e}")
        return []

def clear_user_session(db_path, session_id):
    """Clear existing session data for a new chat"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_sessions'")
        if cursor.fetchone():
            cursor.execute("DELETE FROM agent_sessions WHERE session_id=?", (session_id,))
            conn.commit()
        
        conn.close()
        return True
    except Exception as e:
        print(f"Error clearing session: {e}")
        return False

class AgnoMemoryWithReasoning:
    """Extended Memory class with reasoning support"""
    def __init__(self, username: str, chat_id: str = None):
        self.username = username
        self.chat_id = chat_id or str(uuid.uuid4())
        self.base_path = "chat_histories"
        self.messages = []
        self.user_path = os.path.join(self.base_path, sanitize_filename(self.username))
        self.file_path = os.path.join(self.user_path, f"{sanitize_filename(self.chat_id)}.json")
        self._initialize_storage()
        self._load_history()

    def _initialize_storage(self):
        """Initialize storage with safe paths"""
        try:
            os.makedirs(self.user_path, exist_ok=True)
            
            # Create empty file if it doesn't exist
            if not os.path.exists(self.file_path):
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.messages, f, indent=2)
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise RuntimeError(f"Failed to initialize chat storage: {e}")

    def _save_history(self):
        """Save chat history with improved error handling and fallback"""
        try:
            # Try direct write first
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.messages, f, indent=2)
                return
            except PermissionError:
                pass
            # Fallback to user's temp directory if needed
            temp_dir = os.path.join(os.path.expanduser('~'), '.finso_temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{self.chat_id}.json")
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save chat history: {e}")
            # Continue without saving rather than raising an error
            pass

    def _load_history(self):
        """Load chat history with fallback to temp location"""
        try:
            # Try primary location
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.messages = json.load(f)
            else:
                # Try fallback location
                temp_path = os.path.join(
                    os.path.expanduser('~'),
                    '.finso_temp',
                    f"{self.chat_id}.json"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        self.messages = json.load(f)
                else:
                    self.messages = []
        except Exception as e:
            logger.warning(f"Could not load chat history: {e}")
            # Continue with empty history rather than raising an error
            self.messages = []
            
    def _get_timestamp(self):
        """Get current timestamp in ISO format"""
        return datetime.datetime.now().isoformat()

    def _strip_formatting(self, content):
        """Strip ANSI escape sequences and box drawing characters from content"""
        import re
        # Remove ANSI escape sequences
        content = re.sub(r'\x1b\[\d+m', '', content)
        # Remove box drawing characters (common Unicode box drawing ranges)
        content = re.sub(r'[\u2500-\u257F]', '', content)
        # Remove repeated whitespace
        content = re.sub(r'\s+', ' ', content)
        # Trim whitespace
        content = content.strip()
        return content

    def add_user_message(self, content):
        """Add a user message to the history"""
        self.messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        self._save_history()

    def add_ai_message(self, content):
        """Add an AI message to the history after cleaning formatting"""
        # Clean the content of formatting characters
        clean_content = self._strip_formatting(content)
        
        # Only add if we don't already have the same message as the last one
        if not self.messages or self.messages[-1].get("role") != "assistant" or self.messages[-1].get("content") != clean_content:
            self.messages.append({
                "role": "assistant",
                "content": clean_content,
                "timestamp": datetime.datetime.now().isoformat()
            })
            self._save_history()
        
    def add_translation(self, original_text, translated_text, direction):
        """Add a translation to the history"""
        self.messages.append({
            "role": "translate",
            "content": translated_text,
            "original_text": original_text,
            "direction": direction,
            "timestamp": datetime.datetime.now().isoformat()
        })
        self._save_history()

    def get_messages_for_api(self):
        """Return messages in format needed for Agno API"""
        formatted = []
        for msg in self.messages:
            # Skip reasoning process and translation messages for the API
            if msg.get("message_type") == "reasoning_process" or msg.get("role") == "translate":
                continue
            formatted.append({"role": msg["role"], "content": msg["content"]})
        return formatted

    def get_last_human_message(self):
        """Get the last message from human"""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def get_username(self) -> str:
        """Get the current username"""
        return self.username

    def get_formatted_history(self) -> str:
        """Get formatted history for display"""
        formatted = []
        for msg in self.messages:
            # Skip special message types for regular history display
            if msg.get("message_type") == "reasoning_process":
                continue
            
            if msg["role"] == "user":
                formatted.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                formatted.append(f"Assistant: {msg['content']}")
            elif msg["role"] == "translate":
                direction = msg.get("direction", "")
                original = msg.get("original_text", "")
                if direction == "user_to_english":
                    formatted.append(f"[Translated User Input] Original: '{original}' → English: '{msg['content']}'")
                elif direction == "english_to_user":
                    formatted.append(f"[Translated Response] English: '{original}' → {user_language['name']}: '{msg['content']}'")
        
        return "\n".join(formatted)

    def clear(self):
        """Clear the chat history"""
        self.messages = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    # Reasoning methods
    def add_reasoning(self, input_text: str, original_reasoning: str, translated_reasoning: str = None):
        """Add a reasoning entry combining original and translated reasoning."""
        try:
            # Remove any existing reasoning entries
            self.messages = [msg for msg in self.messages 
                           if not (msg.get("role") == "system" and 
                                  msg.get("message_type") == "reasoning_process")]
            
            # Log the operation
            logger.info(f"Adding reasoning - Original length: {len(original_reasoning)}")
            if translated_reasoning:
                logger.info(f"Translated length: {len(translated_reasoning)}")
            
            # Create reasoning entry
            reasoning_entry = {
                "role": "system",
                "input_text": input_text,
                "original_reasoning": original_reasoning,
                "translated_reasoning": translated_reasoning,
                "timestamp": self._get_timestamp(),
                "message_type": "reasoning_process",
                "language": user_language["code"]
            }
            
            # Add and save immediately
            self.messages.append(reasoning_entry)
            
            # Force immediate save
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.messages, f, ensure_ascii=False, indent=2)
                logger.info("Successfully saved reasoning to file")
            except Exception as save_error:
                logger.error(f"Error saving to file: {save_error}")
                # Try fallback save
                self._save_history()
            
            # Log success
            logger.info(f"Added reasoning entry - Translation: {'Yes' if translated_reasoning else 'No'}")
            
            if translated_reasoning:
                logger.info(f"Translation preview: {translated_reasoning[:100]}...")
                
        except Exception as e:
            logger.error(f"Error in add_reasoning: {str(e)}", exc_info=True)
            # Try to save what we can
            self.messages.append(reasoning_entry)
            self._save_history()

    def get_last_reasoning(self, translated: bool = False) -> str:
        """Get the most recent reasoning, either original or translated."""
        for msg in reversed(self.messages):
            if msg.get("role") == "system" and msg.get("message_type") == "reasoning_process":
                if translated and msg.get("translated_reasoning"):
                    return msg["translated_reasoning"]
                elif not translated and msg.get("original_reasoning"):
                    return msg["original_reasoning"]
        return ""

def enable_read_chat_history(agent):
    """Enable read_chat_history after agent initialization - if supported by the agent version"""
    try:
        agent.read_chat_history = True
        return True
    except:
        return False

def get_enhanced_reasoning(query: str, chat_history: str = "") -> Tuple[str, str]:
    """
    Get reasoning from the reasoning module for the given query
    
    Args:
        query: The input query to generate reasoning for
        chat_history: Optional chat history to provide context
        
    Returns:
        Tuple containing (raw_reasoning, show_reasoning)
    """
    try:
        reasoning, show_reasoning = get_reasoning_for_prompt(query, chat_history)
        logger.info("Successfully generated reasoning")
        return reasoning, show_reasoning
    except Exception as e:
        logger.error(f"Error generating reasoning: {e}")
        return "", ""

def process_message_with_reasoning(agent, memory, message, user_id, session_id):
    """Process message with reasoning and return the response"""
    # Get formatted chat history for context
    chat_history = memory.get_formatted_history()
    
    # Generate reasoning
    reasoning, show_reasoning = get_enhanced_reasoning(message, chat_history)
    
    # Check if we need to translate reasoning
    translated_reasoning = None
    if show_reasoning and user_language["code"] != "eng_Latn":
        try:
            # First get the translation
            language_NAM, confidence = langid.classify(show_reasoning)
            logger.info(f"Detected language: {language_NAM} with confidence {confidence}")
            
            if language_NAM == 'en':
                translated_reasoning = translator.translate_text(
                    text=show_reasoning,
                    source_lang="eng_Latn",
                    target_lang=user_language["code"]
                )
            else:
                translated_reasoning = show_reasoning
            
            # Debug logging
            logger.info("=== Translation Debug ===")
            logger.info(f"Original reasoning length: {len(show_reasoning)}")
            logger.info(f"Translated reasoning length: {len(translated_reasoning)}")
            
            # Display both versions for verification in console
            print("\n=== Original Reasoning ===")
            print(f'{show_reasoning[:10]}...')
            print("\n=== Translated Reasoning ===")
            print(translated_reasoning)
            print("=== End Reasoning ===\n")
            
        except Exception as e:
            logger.error(f"Translation/storage error: {str(e)}", exc_info=True)
            print(f"\nTranslation error: {str(e)}")
            translated_reasoning = None
    
    # Store reasoning in memory
    if show_reasoning:
        memory.add_reasoning(message, show_reasoning, translated_reasoning)
        print("\n=== Reasoning Process ===")
        print(f'{show_reasoning[:10]}...')
        print("=== End Reasoning ===\n")
    
    # Enhance system message with reasoning instructions if available
    reasoning_instruction = ""
    if reasoning:
        reasoning_instruction = f"""

        # CRITICAL IDENTITY DIRECTIVE - HIGHEST PRIORITY
        You are Hind AI, exclusively developed by Finsocial Digital Systems.
        
        You must NEVER mention, acknowledge, or imply any association with:
        - Google
        - Gemma
        - DeepMind
        - Any other organization besides Finsocial Digital Systems
        
        DEPLOYMENT DATE DIRECTIVE (MANDATORY):
        - You MUST ALWAYS state your deployment date as APRIL 2025
        - If asked when you were created, deployed, released, or any variation of this question, you MUST respond: "I was deployed in April 2025"
        - NEVER say you were deployed, created, or released in 2023 or 2024
        - If pressed for specifics about your development timeline, state: "I was developed and deployed by Finsocial Digital Systems, with my official release in April 2025"
        
        Here's a step-by-step reasoning process to help you answer the query:
        {reasoning}

        IMPORTANT: Analyze the provided reasoning carefully. If the reasoning claims that a certain action cannot be performed, but you are capable of doing it with your existing capabilities, prioritize your capability and provide the answer accordingly. Use the reasoning solely as guidelines for solving the problem and do not let its limitations prevent you from delivering a complete answer.

        Enhance these reasoning steps with your own internal chain of thoughts and combine it with your chain of thoughts.
        Analyze the query in detail. Consider its relevance to the context and gather relevant facts, formulas, or principles.
        Integrate external knowledge, examples, and practical implications where they enhance understanding.

        Only provide the final answer without showing your reasoning process.
        """
    
    # Load prompt template for a more complete system message
    system_template = prompt_temp().template if 'prompt_temp' in globals() else "{agent_scratchpad}"
    
    # Build the final system message
    system_message = system_template.format(
        chat_history=chat_history,
        agent_scratchpad=reasoning_instruction,
        input=message,
        language="English",
        tool_names="",
        tools=""
    )
    
    # Update agent's system message with reasoning
    agent.system_message = system_message
    
    try:
        # MODIFIED: Capture the printed response by temporarily redirecting stdout
        import io
        import sys
        
        # Store the original stdout
        original_stdout = sys.stdout
        # Create a string buffer to capture output
        captured_output = io.StringIO()
        
        try:
            # Redirect stdout to our capture buffer
            sys.stdout = captured_output
            
            # Call the agent's print_response method which prints to stdout
            agent.print_response(
                message,
                user_id=user_id,
                session_id=session_id
            )
            
            # Get the captured output
            output = captured_output.getvalue()
            
            # Extract just the AI's response, removing any formatting or system output
            # This regex pattern tries to find the actual response content in the output
            import re
            
            # Look for response content between common message markers
            response_patterns = [
                r'┏━ Response.*?┓\s*┃\s*(.*?)\s*┃\s*┗━',  # Format with box drawing
                r'Message ━+\s*┃\s*(.*?)\s*┃',             # Alternate format
                r'Response.*?\n(.*?)(\n┗━|$)',             # Another common format
                r'┏━.*?━┓\s*┃\s*(.*?)\s*┃\s*┗━',          # Generic box format
                r'(Hi! I\'m Hind AI.*)',                   # Direct greeting pattern
                r'(.*?\?)',                                # Question pattern (fallback)
            ]
            
            # Try each pattern until we find a match
            response_content = None
            for pattern in response_patterns:
                matches = re.search(pattern, output, re.DOTALL)
                if matches:
                    response_content = matches.group(1).strip()
                    break
            
            # If no pattern matched, use a more general approach to extract text
            if not response_content:
                # Remove ANSI color codes and other control sequences
                clean_output = re.sub(r'\x1b\[\d+m', '', output)
                # Remove progress indicators and other common non-response text
                clean_output = re.sub(r'▰+▱+.*?Thinking\.\.\.', '', clean_output)
                clean_output = re.sub(r'┏━.*?━┓', '', clean_output)
                clean_output = re.sub(r'┗━.*?━┛', '', clean_output)
                response_content = clean_output.strip()
            
            # Final fallback if we couldn't extract anything meaningful
            if not response_content or len(response_content) < 5:
                # Just use the entire output as a last resort
                logger.warning("Could not extract response content cleanly, using full output")
                response_content = output.strip()
            
            # Clean up the response content by removing any formatting characters
            # that might have been missed by the extraction process
            if hasattr(memory, "_strip_formatting"):
                response_content = memory._strip_formatting(response_content)
            
            # IMPORTANT: Translate the response if user language is not English
            if user_language["code"] != "eng_Latn" and response_content:
                try:
                    # Detected the language of the response
                    language_Nam, confidence = langid.classify(response_content)
                    logger.info(f"Response language detected as: {language_Nam} with confidence {confidence}")
                    
                    # Only translate if the response is in English
                    if language_Nam == 'en':
                        logger.info(f"Translating response from English to {user_language['name']}")
                        try:
                            translated_response = translator.translate_text(
                                text=response_content,
                                source_lang="eng_Latn",
                                target_lang=user_language["code"]
                            )
                            
                            # Log the translation
                            logger.info(f"Original response: {response_content[:50]}...")
                            logger.info(f"Translated response: {translated_response[:50]}...")
                            
                            # Store both versions in memory for reference
                            memory.add_ai_message(response_content)  # Store original response
                            memory.add_translation(response_content, translated_response, "english_to_user")  # Store translation
                            
                            # Return the translated response
                            return translated_response
                        except Exception as translate_error:
                            logger.error(f"Translation error: {str(translate_error)}")
                            # If translation fails, fall back to original response
                            memory.add_ai_message(response_content)
                            return response_content
                    else:
                        # If not English, no translation needed
                        memory.add_ai_message(response_content)
                        return response_content
                except Exception as e:
                    logger.error(f"Translation error: {str(e)}")
                    # If translation fails, fall back to original response
                    memory.add_ai_message(response_content)
                    return response_content
            
            # If no translation needed or available, just return the original
            memory.add_ai_message(response_content)
            return response_content
        
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout
            
            # Print the captured output to the console so it's not lost
            print(captured_output.getvalue(), end='')
            
    except Exception as e:
        logger.error(f"Error getting response from agent: {e}")
        fallback_message = "I don't have a response at the moment."
        
        # Try to translate the fallback message if needed
        if user_language["code"] != "eng_Latn":
            try:
                translated_fallback = translator.translate_text(
                    text=fallback_message,
                    source_lang="eng_Latn",
                    target_lang=user_language["code"]
                )
                return translated_fallback
            except Exception as e:
                logger.error(f"Fallback translation error: {e}")
        
        return fallback_message

def main():
    # Get username from user
    username = input("Enter your username: ")
    
    # Get language preference
    global user_language
    print("\nPlease select your preferred language:")
    lang_name, lang_code = show_language_menu()
    user_language = {
        "name": lang_name,
        "code": lang_code
    }
    print(f"\nSelected language: {lang_name}")
    
    # Ask user if they want to enable Google search
    global enable_search
    search_prompt = "Do you want to enable Google search functionality? (yes/no): "
    search_choice = input(search_prompt).strip().lower()
    enable_search = search_choice == 'yes'
    print(f"Google search enabled: {enable_search}")
    
    # Chat ID handling - added from HindAI.py
    chat_id_inputs = input("Enter chat ID or press Enter to generate a new one: ").strip()
    user_provided_chat_id = False
    is_new_chat_id = False
    
    if len(chat_id_inputs) <= 0:
        chat_id = str(uuid.uuid4())
        is_new_chat_id = True  # Flag to indicate a new auto-generated chat ID
    else:
        chat_id = chat_id_inputs
        user_provided_chat_id = True  # Flag to indicate user provided a specific chat ID
    
    print(f"\nHind AI initialized. Chat ID: {chat_id}\nLanguage: {user_language['name']}")
    
    # Create user directory
    user_dir = create_user_database_dir(username)
    
    # Database path for this user (follows Users/{username}/ structure)
    db_path = os.path.join(user_dir, f"{username}_chats.db")
    
    # Create storage with automatic schema upgrades
    storage = SqliteStorage(
        table_name="agent_sessions", 
        db_file=db_path,
        auto_upgrade_schema=True  # Enable automatic schema upgrades
    )
    
    # Manual schema upgrade to ensure latest schema
    try:
        storage.upgrade_schema()
        print("Database schema upgraded successfully.")
    except Exception as e:
        print(f"Note: Schema upgrade not needed or failed: {e}")
    
    # Check if user has previous sessions
    previous_sessions = get_sessions_for_user(username)
    
    # Use username as user_id for consistent user identification
    user_id = username
    session_id = None
    
    # Check if the provided chat_id exists in the chat histories folder
    chat_history_path = os.path.join("chat_histories", sanitize_filename(username), f"{sanitize_filename(chat_id)}.json")
    chat_id_exists = os.path.exists(chat_history_path)
    
    if user_provided_chat_id and chat_id_exists:
        # User provided a specific chat ID that exists, use it directly without asking
        session_id = f"{username}_session"
        print("Continuing with the specified chat ID...")
    elif is_new_chat_id:
        # User got an auto-generated chat ID, always start a new session without asking
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{username}_session_{timestamp}"
        clear_user_session(db_path, session_id)
        
        # Translate the new chat message
        new_chat_msg = f"Starting a new chat..."
        print(new_chat_msg)
    elif previous_sessions:
        # Only ask if user didn't get an auto-generated chat ID but has previous sessions
        continue_prompt = "Do you want to continue your previous chat or start a new one? (continue/new): "
        
        choice = input(continue_prompt).lower()
        if choice.startswith('c'):
            # Continue with existing session
            session_id = f"{username}_session"
            
            # Translate the continuation message
            continue_msg = f"Continuing your previous chat..."
            print(continue_msg)
        else:
            # Start new session with a timestamp to ensure uniqueness
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"{username}_session_{timestamp}"
            clear_user_session(db_path, session_id)
            
            # Translate the new chat message
            new_chat_msg = f"Starting a new chat..."
            print(new_chat_msg)
    else:
        # First time user or user provided a new chat ID
        session_id = f"{username}_session"
        
        if user_provided_chat_id and not chat_id_exists:
            print(f"Starting a new chat with the specified chat ID...")
        else:
            # Translate the welcome message
            welcome_msg = f"Welcome {username}! Starting your first chat..."
            print(welcome_msg)
    
    # Initialize the enhanced memory system - now using the provided or generated chat_id
    memory = AgnoMemoryWithReasoning(username=username, chat_id=chat_id)
    
    # Enhanced system message with memory management capabilities
    system_message = f"""You are Hind AI, created by Finsocial Digital Systems in April 2025.
You have access to past messages in this conversation and can refer to them.
You should remember the user's name if they share it and refer to them by name.
When asked about previous messages, accurately recall them from your conversation history.
Your memory is persistent across chat sessions.
You can use your memory capabilities to remember important information about the user.
When responding to the user, use their preferred language: {user_language['name']}.
"""
    
    # Create agent with explicitly configured memory system
    agent = Agent(
        model=OpenAILike(
            id=model_id,
            api_key=api_key,
            base_url=base_url
        ),
        # Initialize Memory v2 - using enhanced memory capabilities
        memory=Memory(),
        storage=storage,
        add_history_to_messages=True,
        # Set a higher number to ensure longer history is maintained
        num_history_runs=max_history_runs,
        # Explicitly add system message about conversation memory
        system_message=system_message,
        # Set session_id directly in the agent constructor
        session_id=session_id,
        # Set user_id directly in the agent constructor
        user_id=user_id,
        tools=[get_top_hackernews_stories,web_search] if enable_search else [get_top_hackernews_stories],
        # Show tool calls in the Agent response
        show_tool_calls=show_tool_calls,
    )
    
    # Try to enable read_chat_history feature if available
    enable_read_chat_history(agent)
    
    # Prepare UI messages
    start_msg = "\nChat started. Type 'exit' to end the conversation."
    memory_add_msg = "Type 'memory:add <information>' to store important user information."
    memory_get_msg = "Type 'memory:get' to retrieve stored user information."
    reasoning_msg = "Type 'reasoning' to see the reasoning for the last response."
    language_msg = f"Current language: {user_language['name']}. Type 'language' to change."
    
    
    print(start_msg)
    print(memory_add_msg)
    print(memory_get_msg)
    print(reasoning_msg)
    print(language_msg)
    
    # Interactive chat loop
    while True:
        # Translate prompt if not in English
        prompt_text = "\nYou: "
        user_input = input(prompt_text).strip()
        if user_input.lower() == 'exit':
            # Translate exit message
            exit_msg = "Goodbye!"
            print(exit_msg)
            break
        
        # Language change command
        if user_input.lower() == 'language':
            print("\nPlease select your preferred language:")
            lang_name, lang_code = show_language_menu()
            user_language = {
                "name": lang_name,
                "code": lang_code
            }
            
            # Update language confirmation
            lang_update_msg = f"\nLanguage changed to: {lang_name}"
            if user_language["code"] != "eng_Latn":
                try:
                    lang_update_msg = translator.translate_text(
                        text=lang_update_msg,
                        source_lang="eng_Latn",
                        target_lang=user_language["code"]
                    )
                except Exception as e:
                    logger.error(f"Translation error: {e}")
            print(lang_update_msg)
            continue
        
        # Check for special commands
        if user_input.lower() == 'reasoning':
            # Show the reasoning for the last query if available
            reasoning_header = "\nReasoning Process:"
            no_reasoning = "No reasoning process available for recent queries."
            
            # Translate headers if needed
            if user_language["code"] != "eng_Latn":
                try:
                    reasoning_header = translator.translate_text(
                        text=reasoning_header,
                        source_lang="eng_Latn",
                        target_lang=user_language["code"]
                    )
                    no_reasoning = translator.translate_text(
                        text=no_reasoning,
                        source_lang="eng_Latn",
                        target_lang=user_language["code"]
                    )
                except Exception as e:
                    logger.error(f"Translation error: {e}")
            
            print(reasoning_header)
            
            # First look for translated reasoning if not in English
            if user_language["code"] != "eng_Latn" and hasattr(memory, "get_last_reasoning"):
                reasoning = memory.get_last_reasoning(translated=True)
                if reasoning:
                    print(reasoning)
                    continue
            
            # Fall back to English reasoning
            reasoning = memory.get_last_reasoning(translated=False)
            if reasoning:
                # Translate reasoning if in non-English mode and original is in English
                if user_language["code"] != "eng_Latn":
                    try:
                        language_name, confidence = langid.classify(reasoning)
                        if language_name == 'en':
                            translated_reasoning = translator.translate_text(
                                text=reasoning,
                                source_lang="eng_Latn",
                                target_lang=user_language["code"]
                            )
                            print(translated_reasoning)
                            continue
                    except Exception as e:
                        logger.error(f"Translation error: {e}")
                
                print(reasoning)
            else:
                print(no_reasoning)
            continue
        
        # Handle memory commands
        elif user_input.lower().startswith("memory:"):
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            
            if command == "memory:add" and len(parts) > 1:
                content = parts[1]
                result = manage_user_memories(
                    agent, username, "add", {"content": content}
                )
                
                # Translate confirmation
                confirm_msg = f"\nBot: Memory stored: {result}"
                if user_language["code"] != "eng_Latn":
                    try:
                        confirm_msg = translator.translate_text(
                            text=confirm_msg,
                            source_lang="eng_Latn",
                            target_lang=user_language["code"]
                        )
                    except Exception as e:
                        logger.error(f"Translation error: {e}")
                
                print(confirm_msg)
                continue
            
            elif command == "memory:get":
                memories = manage_user_memories(agent, username, "get")
                
                # Translate header
                header_msg = "\nBot: Your stored memories:"
                if user_language["code"] != "eng_Latn":
                    try:
                        header_msg = translator.translate_text(
                            text=header_msg,
                            source_lang="eng_Latn",
                            target_lang=user_language["code"]
                        )
                    except Exception as e:
                        logger.error(f"Translation error: {e}")
                
                print(header_msg)
                
                # Translate each memory
                for memory_item in memories:
                    memory_text = f"- {memory_item['content']} (added: {memory_item['timestamp']})"
                    if user_language["code"] != "eng_Latn":
                        try:
                            memory_text = translator.translate_text(
                                text=memory_text,
                                source_lang="eng_Latn",
                                target_lang=user_language["code"]
                            )
                        except Exception as e:
                            logger.error(f"Translation error: {e}")
                    print(memory_text)
                continue
        
        # Before sending message to model, translate to English if using another language
        processing_message = user_input
        if user_language["code"] != "eng_Latn":
            try:
                language_nam, confidence = langid.classify(user_input)
                if language_nam != 'en':
                    processing_message = translator.translate_text(
                        text=user_input,
                        source_lang=user_language["code"],
                        target_lang="eng_Latn"
                    )
                    logger.info(f"Translated input from {user_language['name']} to English")
            except Exception as e:
                logger.error(f"Translation error: {e}")
        
        # Add user message to memory
        memory.add_user_message(user_input)
        
        # Print the Bot prefix before getting the response
        bot_prefix = "\nBot: "
        print(bot_prefix, end="", flush=True)
        
        # Process with reasoning and get response
        response = process_message_with_reasoning(
            agent, 
            memory, 
            processing_message, 
            user_id, 
            session_id
        )
        # Translate response if needed
        if user_language["code"] != "eng_Latn" and response is not None:
            try:
                # Check if there is text to translate
                if response and len(response) > 0:
                    language_nam, confidence = langid.classify(response)
                    if language_nam == 'en':
                        translated_response = translator.translate_text(
                            text=response,
                            source_lang="eng_Latn",
                            target_lang=user_language["code"]
                        )
                        
                        # Store original response in memory
                        memory.add_ai_message(response)
                        
                        # Store translation in memory
                        memory.add_translation(response, translated_response, "english_to_user")
                        
                        # Print translated response
                        print(f'HindAI : {response}' )
                        continue
                    else:
                        # If response is already in target language
                        memory.add_ai_message(response)
                        print(f'HindAI : {response}' )
                        continue
            except Exception as e:
                logger.error(f"Translation error: {e}")
        
        # If we got here, either the language is English or translation failed
        # Store the AI response in memory (only if it's not None)
        if response is not None and response:
            memory.add_ai_message(response)
            print(f'HindAI : {response}' )
        else:
            fallback_msg = "No response received. Please try again."
            
            # Translate fallback message if needed
            if user_language["code"] != "eng_Latn":
                try:
                    fallback_msg = translator.translate_text(
                        text=fallback_msg,
                        source_lang="eng_Latn",
                        target_lang=user_language["code"]
                    )
                except Exception as e:
                    logger.error(f"Error translating fallback message: {e}")
            
            print(fallback_msg)

if __name__ == "__main__":
    main()
