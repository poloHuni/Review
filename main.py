import streamlit as st
import requests
import json
import time
import os
import base64
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import uuid
from audio_recorder_streamlit import audio_recorder
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re
import extra_streamlit_components as stx

# Constants
OWNER_EMAIL = "natnaelgebremichaeltewelde@gmail.com"
BRAND_COLOR = "#f8f9fa"
ACCENT_COLOR = "#e53935"
LIGHT_COLOR = "#5a7d7c"

# Load environment variables
load_dotenv()

# Database functions (unchanged)
def load_reviews_from_db(limit=None):
    try:
        if os.path.exists("restaurant_reviews.xlsx"):
            df = pd.read_excel("restaurant_reviews.xlsx")
            reviews = df.to_dict(orient='records')
            # Sort and apply limit only if specified
            if limit is not None and reviews:
                reviews = sorted(reviews, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
            return reviews
        else:
            # Create empty file if doesn't exist
            df = pd.DataFrame(columns=[
                "timestamp", "customer_id", "customer_name", "customer_email", 
                "customer_phone", "summary", "food_quality", "service", 
                "atmosphere", "sentiment_score", "specific_points", 
                "improvement_suggestions", "raw_transcription"
            ])
            df.to_excel("restaurant_reviews.xlsx", index=False)
            return []
    except Exception as e:
        st.error(f"Error loading reviews database: {str(e)}")
        return []

def save_review(review_data):
    # Add metadata
    review_data.update({
        "timestamp": datetime.now().isoformat(),
        "customer_id": st.session_state.customer_id,
        "customer_name": st.session_state.customer_info["name"],
        "customer_email": st.session_state.customer_info["email"],
        "customer_phone": st.session_state.customer_info["phone"]
    })
    
    try:
        # Load existing reviews
        all_reviews = load_reviews_from_db()
        all_reviews.append(review_data)
        
        # Convert to DataFrame
        df_reviews = pd.DataFrame(all_reviews)
        
        # Handle nested lists
        for col in df_reviews.columns:
            if len(df_reviews) > 0 and isinstance(df_reviews[col].iloc[0], list):
                df_reviews[col] = df_reviews[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Save to Excel
        df_reviews.to_excel("restaurant_reviews.xlsx", index=False)
        st.session_state.reviews = all_reviews
        return True
    except Exception as e:
        st.error(f"Error saving review: {str(e)}")
        return False

# Function to get the cookie manager
def get_cookie_manager():
    """
    Singleton pattern for cookie manager to avoid duplicate key errors.
    Returns a single instance of the CookieManager across the application.
    """
    # Check if we already have a cookie manager in session state
    if 'cookie_manager' not in st.session_state:
        # Create with a unique key
        st.session_state.cookie_manager = stx.CookieManager(key="unique_cookie_manager")
    
    # Return the stored instance
    return st.session_state.cookie_manager

# User management functions (unchanged)
def load_users_from_db():
    try:
        if os.path.exists("restaurant_users.xlsx"):
            return pd.read_excel("restaurant_users.xlsx").to_dict(orient='records')
        else:
            # Create empty file if doesn't exist
            df = pd.DataFrame(columns=[
                "user_id", "username", "email", "phone", "name", 
                "last_login", "registration_date"
            ])
            df.to_excel("restaurant_users.xlsx", index=False)
            return []
    except Exception as e:
        st.error(f"Error loading users database: {str(e)}")
        return []

def save_user(user_data):
    try:
        # Load existing users
        all_users = load_users_from_db()
        
        # Check if user already exists
        existing_user = next((user for user in all_users if user.get('user_id') == user_data.get('user_id')), None)
        
        if existing_user:
            # Update existing user
            for key, value in user_data.items():
                existing_user[key] = value
        else:
            # Add new user
            all_users.append(user_data)
        
        # Convert to DataFrame
        df_users = pd.DataFrame(all_users)
        
        # Save to Excel
        df_users.to_excel("restaurant_users.xlsx", index=False)
        return True
    except Exception as e:
        st.error(f"Error saving user: {str(e)}")
        return False

def find_user_by_email(email):
    # Normalize email to lowercase
    normalized_email = email.lower().strip()
    users = load_users_from_db()
    return next((user for user in users if user.get('email', '').lower().strip() == normalized_email), None)

def validate_sa_phone_number(phone):
    """
    Validate a South African phone number.
    Returns: (is_valid, formatted_number)
    """
    import re
    
    if not phone or phone.strip() == "":
        return True, ""  # Empty phone is allowed
        
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # South African numbers should be 10 digits (excluding country code)
    if len(digits_only) == 10 and digits_only.startswith('0'):
        # Format as 0XX XXX XXXX
        formatted = f"{digits_only[0:3]} {digits_only[3:6]} {digits_only[6:10]}"
        return True, formatted
    
    # Check if it has international code (+27)
    elif len(digits_only) == 11 and digits_only.startswith('27'):
        # Convert to local format (0XX XXX XXXX)
        formatted = f"0{digits_only[2:4]} {digits_only[4:7]} {digits_only[7:11]}"
        return True, formatted
    
    # Check if it has international code with + (+27)
    elif len(digits_only) == 12 and digits_only.startswith('270'):
        # Convert to local format (0XX XXX XXXX)
        formatted = f"0{digits_only[3:5]} {digits_only[5:8]} {digits_only[8:12]}"
        return True, formatted
        
    return False, None

def get_user_reviews(user_id, limit=None):
    try:
        if os.path.exists("restaurant_reviews.xlsx"):
            df = pd.read_excel("restaurant_reviews.xlsx")
            # Filter reviews by user_id
            user_reviews = df[df['customer_id'] == user_id].to_dict(orient='records')
            
            # Sort and limit if specified
            if limit is not None and user_reviews:
                user_reviews = sorted(user_reviews, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
                
            return user_reviews
        return []
    except Exception as e:
        st.error(f"Error loading user reviews: {str(e)}")
        return []

# Update these functions in your main application
def init_session_state():
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'customer_id' not in st.session_state:
        st.session_state.customer_id = str(uuid.uuid4())
    if 'reviews' not in st.session_state:
        st.session_state.reviews = load_reviews_from_db()
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'show_analysis' not in st.session_state:
        st.session_state.show_analysis = False
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'record_again' not in st.session_state:
        st.session_state.record_again = False
    if 'customer_info' not in st.session_state:
        st.session_state.customer_info = {"name": "", "email": "", "phone": ""}
    # New session state variables for login system
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'login_error' not in st.session_state:
        st.session_state.login_error = None
    if 'register_success' not in st.session_state:
        st.session_state.register_success = None
    # Add the is_owner flag
    if 'is_owner' not in st.session_state:
        st.session_state.is_owner = False
        
# Login form - Enhanced with better styling
def render_login_form():
    st.markdown(f"""
    <div class="login-container">
        <h2 style="color: #FFFFFF; text-align: center; margin-bottom: 30px;">
            <span style="font-size: 32px;">👋</span> Welcome to Our Feedback Portal
        </h2>
        
        <div class="login-section">
            <h3 style="color: #FFFFFF; font-size: 22px; margin-bottom: 15px;">Login</h3>
            <p style="margin-bottom: 20px; color: #CCCCCC;">Access your account to view your previous feedback and submit new reviews.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a card-like container for the login form
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            with st.form("login_form", border=False):
                email = st.text_input("Email Address", key="login_email")
                submit_login = st.form_submit_button("Login", use_container_width=True)
                
                if submit_login:
                    # Normalize email to lowercase for case-insensitive comparison
                    normalized_email = email.lower().strip()
                    user = find_user_by_email(normalized_email)
                    cookie_manager = get_cookie_manager()

                    if user:
                        # Set cookie
                        cookie_manager.set(
                            cookie="user_id", 
                            val=user.get('user_id'),
                            expires_at=datetime.now() + timedelta(days=30)
                        )            
                        # Update session state
                        st.session_state.is_logged_in = True
                        st.session_state.current_user = user
                        st.session_state.customer_id = user.get('user_id')
                        st.session_state.customer_info = {
                            "name": user.get('name', ""),
                            "email": user.get('email', ""),
                            "phone": user.get('phone', "")
                        }
                        
                        # Check if this user is the owner
                        st.session_state.is_owner = (user.get('email', "").lower() == OWNER_EMAIL.lower())
                        
                        # Update last login time
                        user['last_login'] = datetime.now().isoformat()
                        save_user(user)
                        
                        st.rerun()
                    else:
                        st.session_state.login_error = "User not found. Please register first."
    
    if st.session_state.login_error:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.error(st.session_state.login_error)
    
    # Registration section
    st.markdown(f"""
    <div class="login-container" style="margin-top: 30px;">
        <div class="register-section">
            <h3 style="color: #FFFFFF; font-size: 22px; margin-bottom: 15px;">New User? Register Here</h3>
            <p style="margin-bottom: 20px; color: #CCCCCC;">Create an account to start sharing your dining experiences with us.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.form("register_form", border=False):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            phone = st.text_input("Phone (South African format: 0XX XXX XXXX)")
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                # Normalize email
                normalized_email = email.lower().strip()
                
                # Check if user already exists (case insensitive)
                existing_user = find_user_by_email(normalized_email)
                
                if existing_user:
                    st.error("A user with this email already exists. Please login instead.")
                elif not email or not name:
                    st.error("Name and email are required.")
                else:
                    # Validate phone number
                    is_valid_phone, formatted_phone = validate_sa_phone_number(phone)
                    
                    if phone and not is_valid_phone:
                        st.error("Please enter a valid South African phone number (e.g., 071 123 4567 or +27 71 123 4567)")
                    else:
                        # Create new user
                        user_id = str(uuid.uuid4())
                        new_user = {
                            "user_id": user_id,
                            "username": normalized_email.split('@')[0],
                            "email": normalized_email,  # Store normalized email
                            "phone": formatted_phone,   # Store formatted phone
                            "name": name,
                            "last_login": datetime.now().isoformat(),
                            "registration_date": datetime.now().isoformat()
                        }
                        
                        if save_user(new_user):
                            # Set cookie
                            cookie_manager = get_cookie_manager()
                            cookie_manager.set(
                                cookie="user_id", 
                                val=user_id,
                                expires_at=datetime.now() + timedelta(days=30)
                            )                    
                            # Update session state
                            st.session_state.is_logged_in = True
                            st.session_state.current_user = new_user
                            st.session_state.customer_id = user_id
                            st.session_state.customer_info = {
                                "name": name,
                                "email": normalized_email,
                                "phone": formatted_phone
                            }
                            st.session_state.register_success = "Registration successful!"
                            st.rerun()
                        else:
                            st.error("Error during registration. Please try again.")

# Check if user is logged in via cookie
def check_login_status():
    if st.session_state.is_logged_in:
        return True
        
    cookie_manager = get_cookie_manager()
    user_id = cookie_manager.get("user_id")
    
    if user_id:
        # Find user in database
        users = load_users_from_db()
        user = next((u for u in users if u.get('user_id') == user_id), None)
        
        if user:
            # Update session state
            st.session_state.is_logged_in = True
            st.session_state.current_user = user
            st.session_state.customer_id = user.get('user_id')
            st.session_state.customer_info = {
                "name": user.get('name', ""),
                "email": user.get('email', ""),
                "phone": user.get('phone', "")
            }
            
            # Check if this user is the owner (case-insensitive comparison)
            st.session_state.is_owner = (user.get('email', '').lower() == OWNER_EMAIL.lower())
            
            # Update last login time
            user['last_login'] = datetime.now().isoformat()
            save_user(user)
            
            return True
    
    return False

# Logout function
def logout():
    cookie_manager = get_cookie_manager()
    cookie_manager.delete("user_id")
    
    # Reset session state
    st.session_state.is_logged_in = False
    st.session_state.current_user = None
    st.session_state.customer_id = str(uuid.uuid4())
    st.session_state.customer_info = {"name": "", "email": "", "phone": ""}
    st.session_state.is_owner = False
    
    st.rerun()

# LLM functions (unchanged)
def get_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.4, streaming=True)

def initialize_conversation():
    if st.session_state.conversation is None:
        llm = get_llm()
        st.session_state.conversation = ConversationChain(
            llm=llm, memory=st.session_state.memory, verbose=True
        )

def validate_review_input(text):
    if len(text.strip()) < 10:
        return False, "Your feedback seems too short. Please provide more details about your experience."
    
    initialize_conversation()
    
    validation_prompt = f"""
    You need to determine if the following text is a valid restaurant review/feedback or not.
    
    Text: "{text}"
    
    A valid restaurant review typically mentions food, service, atmosphere, or specific experiences at a restaurant.
    Greetings, questions about the system, or unrelated conversations are NOT valid reviews.
    
    Return a JSON object with this format:
    {{
        "is_valid": true/false,
        "reason": "Brief explanation of why it is or isn't a valid review",
        "review_elements": ["food", "service", "atmosphere", "experience"] (include only those that are present)
    }}
    
    Respond with ONLY the JSON object, no additional text.
    """
    
    response = st.session_state.conversation.predict(input=validation_prompt)
    
    try:
        # Extract JSON
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, response)
        
        if match:
            validation_result = json.loads(match.group(1))
        else:
            validation_result = json.loads(response)
        
        is_valid = validation_result.get('is_valid', False)
        reason = validation_result.get('reason', "")
        
        return (is_valid, None) if is_valid else (False, reason)
    except Exception:
        return False, "Unable to validate your feedback. Please ensure you're providing restaurant-related feedback."

def process_and_validate_review(text):
    if not text:
        return None, "Please provide some feedback."
    
    is_valid, validation_message = validate_review_input(text)
    
    if not is_valid:
        return None, validation_message
    
    return process_review(text), None

# Audio functions
def process_audio_file(input_file_path):
    """Apply audio processing to improve quality before transcription"""
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
        
        # Load the audio file
        audio = AudioSegment.from_file(input_file_path, format="wav")
        
        # Normalize the volume (makes quiet parts louder and loud parts quieter)
        normalized_audio = normalize(audio)
        
        # Apply noise reduction (simple high-pass filter to reduce low-frequency noise)
        filtered_audio = normalized_audio.high_pass_filter(80)
        
        # Export the processed file
        processed_file_path = input_file_path.replace(".wav", "_processed.wav")
        filtered_audio.export(processed_file_path, format="wav")
        
        return processed_file_path
    except Exception as e:
        st.warning(f"Audio processing failed, using original audio: {str(e)}")
        return input_file_path

def transcribe_audio(audio_file_path):
    try:
        # Get API token from environment variable
        LELAPA_API_TOKEN = os.environ.get("LELAPA_API_TOKEN")
        
        if not LELAPA_API_TOKEN:
            st.error("Lelapa API token not found. Please add it to your .env file.")
            return None

        # Open file in binary mode for proper upload
        with open(audio_file_path, 'rb') as file:
            files = {"file": file}
            
            # Set up headers with client token
            headers = {
                "X-CLIENT-TOKEN": LELAPA_API_TOKEN,
            }
            
            # Make API request to the new endpoint
            response = requests.post(
                "https://vulavula-services.lelapa.ai/api/v2alpha/transcribe/sync/file",
                files=files,
                headers=headers,
            )
        
        # Check response status
        if response.status_code == 200:
            response_json = response.json()
            
            # Extract transcription text from response
            transcription_text = response_json.get("transcription_text", response_json.get("transcription", ""))
            
            return transcription_text
        else:
            st.error(f"Error transcribing audio: Status code {response.status_code}")
            st.write(f"Response: {response.text}")
            return None

    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def record_audio():
    instruction_container = st.empty()
    
    # Add recording tips with improved styling and dark theme
    st.markdown(f"""
    <div style="background-color: #333333; border-radius: 10px; padding: 20px; margin-bottom: 20px; border-left: 5px solid {BRAND_COLOR};">
        <h4 style="color: #FFFFFF; margin-top: 0; font-size: 18px;">📝 Tips for Better Audio Quality</h4>
        <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
            <li style="margin-bottom: 8px;">Speak clearly and at a normal pace</li>
            <li style="margin-bottom: 8px;">Keep the microphone 4-6 inches from your mouth</li>
            <li style="margin-bottom: 8px;">Reduce background noise when possible</li>
            <li style="margin-bottom: 8px;">Ensure your device microphone is not covered</li>
            <li style="margin-bottom: 0;">Use headphones with a built-in microphone for better results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    recorder_container = st.empty()
    process_container = st.empty()
    
    # Handle "Record Again" flow
    if st.session_state.record_again:
        st.session_state.record_again = False
        st.rerun()
    
    # Show instruction with dark theme styling
    instruction_container.markdown(f"""
    <div style="background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 20px; 
         border-left: 5px solid {BRAND_COLOR}; display: flex; align-items: center;">
        <div style="font-size: 24px; margin-right: 15px;">🎙️</div>
        <div>
            <p style="margin: 0; font-size: 16px; color: #E0E0E0;">Click the microphone to start recording and click again to stop recording.</p>
            <p style="margin: 0; font-size: 14px; color: #BBBBBB; margin-top: 5px;">Maximum recording time: 25 seconds</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Only show recorder if we don't have audio already recorded
    if not hasattr(st.session_state, 'audio_file') or st.session_state.audio_file is None:
        # Show recorder
        with recorder_container:
            audio_bytes = audio_recorder(
                text="Press and hold to record",
                recording_color=ACCENT_COLOR,
                neutral_color=BRAND_COLOR,
                icon_name="microphone",
                pause_threshold=25.0,
                energy_threshold=0.01,  # Lower threshold for mobile mics
                sample_rate=48000      # Standard sample rate
            )
        
        # Process recorded audio
        if audio_bytes:
            # Immediately update UI to show recording completed and waiting message
            instruction_container.markdown(f"""
            <div style="background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 20px; 
                border-left: 5px solid #4caf50; display: flex; align-items: center;">
                <div style="font-size: 24px; margin-right: 15px;">✅</div>
                <div>
                    <p style="margin: 0; font-size: 16px; font-weight: 500; color: #E0E0E0;">Recording completed!</p>
                    <p style="margin: 0; font-size: 14px; color: #BBBBBB; margin-top: 5px;">Please wait while we prepare your options...</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if the recording has enough data
            if len(audio_bytes) < 1000:  # If recording is too short (less than ~0.1 seconds)
                instruction_container.markdown(f"""
                <div style="background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 20px; 
                    border-left: 5px solid #ffc107; display: flex; align-items: center;">
                    <div style="font-size: 24px; margin-right: 15px;">⚠️</div>
                    <div>
                        <p style="margin: 0; font-size: 16px; font-weight: 500; color: #E0E0E0;">Recording was too short</p>
                        <p style="margin: 0; font-size: 14px; color: #BBBBBB; margin-top: 5px;">Please click the mic button to re-record. Wait for 5 seconds after stopping recording for next steps.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Save audio file in background with proper directory and safe filename
                audio_dir = ensure_directories()
                timestamp = generate_safe_timestamp()
                filename = f"{audio_dir}/review_{st.session_state.customer_id}_{timestamp}.wav"
                
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                
                # Update session state
                st.session_state.audio_file = filename
                
                # Save the recording and rerun to display the buttons
                st.rerun()
    else:
        # We have a recording, show buttons and hide recorder
        instruction_container.markdown(f"""
        <div style="background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 20px; 
            border-left: 5px solid #4caf50; display: flex; align-items: center;">
            <div style="font-size: 24px; margin-right: 15px;">✅</div>
            <div>
                <p style="margin: 0; font-size: 16px; font-weight: 500; color: #E0E0E0;">Recording completed!</p>
                <p style="margin: 0; font-size: 14px; color: #BBBBBB; margin-top: 5px;">Ready for next steps.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process button with dark theme styling
        with process_container:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Process Recording", key="process_audio_btn", 
                           use_container_width=True, 
                           type="primary"):
                    with st.spinner("Processing and enhancing your audio..."):
                        # Apply audio processing to improve quality
                        processed_audio_file = process_audio_file(st.session_state.audio_file)
                        
                    with st.spinner("Transcribing your feedback..."):
                        transcribed_text = transcribe_audio(processed_audio_file)
                        
                    if transcribed_text:
                        with st.spinner("Analyzing your feedback..."):
                            review_analysis = process_review(transcribed_text)
                            
                        if review_analysis:
                            st.session_state.current_analysis = review_analysis
                            st.session_state.show_analysis = True
                            st.rerun()
                        else:
                            st.error("Failed to analyze your feedback. Please try again.")
                    else:
                        st.error("Failed to transcribe your audio. Please try again.")
            
            # Record again button
            with col2:
                if st.button("🔄 Record Again", key="record_again_btn", 
                           use_container_width=True):
                    if hasattr(st.session_state, 'audio_file') and st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                        try:
                            os.remove(st.session_state.audio_file)
                        except Exception as e:
                            # Just log the error but don't crash
                            print(f"Error removing audio file: {str(e)}")
                            
                    # Reset states
                    st.session_state.audio_file = None
                    st.session_state.show_analysis = False
                    st.session_state.current_analysis = None
                    st.session_state.is_recording = False
                    st.session_state.record_again = True
                    st.rerun()
    
    return None
def process_review(transcribed_text):
    if not transcribed_text:
        return None
    
    initialize_conversation()
    
    prompt = f"""
    You are analyzing a customer's feedback for a bar and restaurant that features live DJs and music. 
    Please summarize this feedback, extracting key points about the overall experience, food quality, 
    service, atmosphere (including music and entertainment), and any specific recommendations or complaints. 
    Also, rate the overall sentiment on a scale of 1-5.
    
    Customer feedback: {transcribed_text}
    
    Provide your analysis in the following JSON format. Make sure to escape any quotes within the text fields:
    {{
        "summary": "Brief summary of the overall experience",
        "food_quality": "Assessment of food and drinks",
        "service": "Assessment of service quality",
        "atmosphere": "Assessment of ambiance, music, and entertainment",
        "music_and_entertainment": "Specific feedback on DJs, music selection, and overall vibe",
        "specific_points": ["Point 1", "Point 2", "..."],
        "sentiment_score": X,
        "improvement_suggestions": ["Suggestion 1", "Suggestion 2", "..."]
    }}
    
    Important: Ensure your response is ONLY the valid JSON object, with no additional text before or after.
    """
    
    response = st.session_state.conversation.predict(input=prompt)
    
    try:
        # Extract JSON more robustly
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, response)
        
        if match:
            json_str = match.group(1)
            # Try to fix common JSON errors
            json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
            json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)  # Add quotes to keys
            parsed_response = json.loads(json_str)
        else:
            # Fallback to direct loading if regex doesn't match
            parsed_response = json.loads(response)
            
        # Add raw transcription
        parsed_response["raw_transcription"] = transcribed_text
        return parsed_response
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse LLM response as JSON: {str(e)}")
        # Create a basic response with the transcription
        return {
            "summary": "Error processing review. Please try again.",
            "food_quality": "N/A",
            "service": "N/A",
            "atmosphere": "N/A",
            "music_and_entertainment": "N/A",
            "specific_points": ["Unable to process specific points"],
            "sentiment_score": 3,  # Neutral default
            "improvement_suggestions": ["Unable to process improvement suggestions"],
            "raw_transcription": transcribed_text,
            "error": f"JSON parsing error: {str(e)}"
        }

# UI functions - Enhanced with better styling
def display_analysis(review_analysis):
    with st.container():
        st.markdown(f"""
        <div style="padding: 25px; border-radius: 10px; border: 1px solid #444444; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); background-color: #2D2D2D; margin-bottom: 20px;">
            <h3 style="color: #FFFFFF; margin-top: 0; margin-bottom: 20px; font-size: 22px; 
                border-bottom: 2px solid {BRAND_COLOR}; padding-bottom: 10px;">
                Feedback Analysis
            </h3>
        """, unsafe_allow_html=True)
        
        # Summary section with improved styling
        st.markdown(f"""
        <div style="margin-bottom: 20px; background-color: #333333; padding: 15px; border-radius: 8px;">
            <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 18px;">Summary</h4>
            <p style="margin: 0; color: #E0E0E0;">{review_analysis.get('summary', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a grid for the assessment categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #333333; padding: 15px; border-radius: 8px; height: 100%;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                    <span style="margin-right: 8px;">🍽️</span> Food Quality
                </h4>
                <p style="margin: 0; color: #E0E0E0;">{review_analysis.get('food_quality', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div style="background-color: #333333; padding: 15px; border-radius: 8px; height: 100%;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                    <span style="margin-right: 8px;">👨‍🍳</span> Service
                </h4>
                <p style="margin: 0; color: #E0E0E0;">{review_analysis.get('service', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-top: 15px; height: 100%;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                    <span style="margin-right: 8px;">🏮</span> Atmosphere
                </h4>
                <p style="margin: 0; color: #E0E0E0;">{review_analysis.get('atmosphere', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-top: 15px; height: 100%;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                    <span style="margin-right: 8px;">🎵</span> Music & Entertainment
                </h4>
                <p style="margin: 0; color: #E0E0E0;">{review_analysis.get('music_and_entertainment', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment indicator with animated stars
        sentiment = review_analysis.get('sentiment_score', 'N/A')
        if isinstance(sentiment, (int, float)):
            st.markdown(f"""
            <div style="margin-top: 20px; background-color: #333333; padding: 15px; border-radius: 8px; 
                text-align: center;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 18px;">
                    Overall Sentiment
                </h4>
                {display_animated_stars(sentiment)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-top: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 8px; 
                 text-align: center;">
                <h4 style="color: {BRAND_COLOR}; margin-top: 0; margin-bottom: 10px; font-size: 18px;">
                    Overall Sentiment
                </h4>
                <p style="margin: 0; font-size: 18px;">{sentiment}/5</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display key points with improved styling
        if 'specific_points' in review_analysis:
            st.markdown(f"""
            <div style="margin-top: 20px; background-color: #333333; padding: 15px; border-radius: 8px;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 18px;">
                    <span style="margin-right: 8px;">🔑</span> Key Points
                </h4>
                <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
            """, unsafe_allow_html=True)
            
            for point in review_analysis['specific_points']:
                st.markdown(f"<li style='margin-bottom: 8px;'>{point}</li>", unsafe_allow_html=True)
                
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Display suggestions with improved styling
        if 'improvement_suggestions' in review_analysis:
            st.markdown(f"""
            <div style="margin-top: 20px; background-color: #333333; padding: 15px; border-radius: 8px;">
                <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 18px;">
                    <span style="margin-right: 8px;">💡</span> Suggestions for Improvement
                </h4>
                <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
            """, unsafe_allow_html=True)
            
            for suggestion in review_analysis['improvement_suggestions']:
                st.markdown(f"<li style='margin-bottom: 8px;'>{suggestion}</li>", unsafe_allow_html=True)
                
            st.markdown("</ul></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def collect_customer_info():
    # If the user is logged in, display their info without ability to edit
    if st.session_state.is_logged_in:
        st.markdown(f"""
        <div style="background-color: #333333; padding: 20px; border-radius: 10px; margin-bottom: 20px; 
            border-left: 5px solid {BRAND_COLOR};">
            <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 15px; font-size: 18px;">
                <span style="margin-right: 8px;">👤</span> Your Information
            </h4>
            <div style="display: flex; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #BBBBBB; font-size: 14px;">Name</p>
                    <p style="margin: 0; font-weight: 500; font-size: 16px; color: #FFFFFF;">{st.session_state.customer_info['name']}</p>
                </div>
                <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #BBBBBB; font-size: 14px;">Email</p>
                    <p style="margin: 0; font-weight: 500; font-size: 16px; color: #FFFFFF;">{st.session_state.customer_info['email']}</p>
                </div>
                <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #BBBBBB; font-size: 14px;">Phone</p>
                    <p style="margin: 0; font-weight: 500; font-size: 16px; color: #FFFFFF;">{st.session_state.customer_info['phone'] or 'Not provided'}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add edit button that redirects to account settings (optional)
        if st.button("✏️ Edit Profile", type="secondary"):
            st.info("To edit your profile information, please contact restaurant management.")
            
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # For non-logged in users, allow editing with improved styling
        st.markdown(f"""
        <div style="background-color: #333333; padding: 20px; border-radius: 10px; margin-bottom: 20px; 
            border-left: 5px solid {BRAND_COLOR};">
            <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 15px; font-size: 18px;">
                <span style="margin-right: 8px;">👤</span> Your Information
            </h4>
            <p style="margin-bottom: 15px; color: #E0E0E0;">Please provide your contact information to submit feedback.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            customer_name = st.text_input("Name", value=st.session_state.customer_info["name"])
        with col2:
            customer_email = st.text_input("Email", value=st.session_state.customer_info["email"])
        
        customer_phone = st.text_input("Phone (optional)", value=st.session_state.customer_info["phone"])
        
        # Update customer info
        st.session_state.customer_info = {
            "name": customer_name, "email": customer_email, "phone": customer_phone
        }

# CSS for styling - Enhanced with modern design elements
def load_css():
    st.markdown(f"""
    <style>
    /* Page background and text */
    .stApp {{
        background-color: #1E1E1E;  /* Dark background */
        color: #E0E0E0;  /* Light text */
    }}
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #FFFFFF;  /* White headers */
    }}
    
    h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {BRAND_COLOR};
    }}
    
    h2 {{
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
    }}
    
    /* Custom styling for tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: #2D2D2D;  /* Darker background for tabs */
        border-radius: 10px;
        padding: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: #2D2D2D;
        border-radius: 8px;
        color: #CCCCCC;  /* Light gray text */
        padding: 10px 16px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {BRAND_COLOR};
        color: white;
    }}
    
    /* Card-like container style */
    .card-container {{
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #444444;  /* Darker border */
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        background-color: #2D2D2D;  /* Dark card background */
        margin-bottom: 20px;
    }}
    
    /* Custom button styling */
    .stButton>button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }}
    
    /* Primary button */
    .stButton>button[data-baseweb="button"][kind="primary"] {{
        background-color: {BRAND_COLOR};
    }}
    
    .stButton>button[data-baseweb="button"][kind="primary"]:hover {{
        background-color: #4a6d6c;
    }}
    
    /* Secondary button */
    .stButton>button[data-baseweb="button"]:not([kind="primary"]) {{
        border: 1px solid #444444;
        background-color: #333333;
        color: #E0E0E0;
    }}
    
    .stButton>button[data-baseweb="button"]:not([kind="primary"]):hover {{
        border-color: {BRAND_COLOR};
        color: {BRAND_COLOR};
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        font-size: 16px;
        background-color: #2D2D2D;  /* Darker background */
        border-radius: 8px;
        padding: 12px 15px;
        font-weight: 500;
        color: #E0E0E0;  /* Light text */
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: #333333;
    }}
    
    .streamlit-expanderContent {{
        border: none;
        border-top: 1px solid #444444;
        padding: 15px;
        background-color: #2D2D2D;
    }}
    
    /* Text area styling */
    .stTextArea>div>div {{
        border-radius: 8px;
        border-color: #444444;
        background-color: #333333;
        color: #E0E0E0;
    }}
    
    .stTextArea textarea {{
        color: #E0E0E0;
        background-color: #333333;
    }}
    
    .stTextArea textarea:focus {{
        border-color: {BRAND_COLOR};
        box-shadow: 0 0 0 1px {BRAND_COLOR};
    }}
    
    /* Text input styling */
    .stTextInput>div>div {{
        border-radius: 8px;
        border-color: #444444;
        background-color: #333333;
    }}
    
    .stTextInput input {{
        color: #E0E0E0;
        background-color: #333333;
    }}
    
    .stTextInput input:focus {{
        border-color: {BRAND_COLOR};
        box-shadow: 0 0 0 1px {BRAND_COLOR};
    }}
    
    /* Form styling */
    [data-testid="stForm"] {{
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    /* Login container styling */
    .login-container {{
        max-width: 600px;
        margin: 0 auto;
        padding: 30px;
        border-radius: 10px;
        border: 1px solid #444444;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: #2D2D2D;
    }}
    
    .login-header {{
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 30px;
    }}
    
    .form-divider {{
        margin: 25px 0;
        border-top: 1px solid #444444;
    }}
    
    /* Star rating animation */
    @keyframes star-pop {{
        0% {{ transform: scale(0.8); opacity: 0.4; }}
        50% {{ transform: scale(1.2); opacity: 0.9; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    
    @keyframes star-glow {{
        0% {{ text-shadow: 0 0 0px rgba(255, 215, 0, 0); }}
        50% {{ text-shadow: 0 0 8px rgba(255, 215, 0, 0.8); }}
        100% {{ text-shadow: 0 0 4px rgba(255, 215, 0, 0.4); }}
    }}
    
    .animated-star {{
        display: inline-block;
        animation: star-pop 0.3s ease-out forwards, star-glow 1.5s ease-in-out infinite alternate;
        animation-delay: calc(var(--star-index) * 0.1s), calc(var(--star-index) * 0.1s + 0.3s);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: #252525;
        border-right: 1px solid #444444;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {{
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #444444;
    }}
    
    /* Spinner styling */
    .stSpinner > div {{
        border-color: {BRAND_COLOR} !important;
    }}
    
    /* Alert styling */
    .stAlert {{
        border-radius: 8px;
        border: none;
    }}
    
    .stAlert [data-baseweb="notification"] {{
        border-radius: 8px;
        border: none;
    }}
    
    /* Selectbox styling */
    .stSelectbox label,
    .stDateInput label {{
        color: #E0E0E0;
    }}
    
    .stSelectbox > div > div,
    .stDateInput > div > div {{
        background-color: #333333;
        border-color: #444444;
    }}
    
    /* Make markdown text light */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] li, 
    [data-testid="stMarkdownContainer"] span {{
        color: #E0E0E0;
    }}
    
    /* Fix info/error/warning boxes */
    .stAlert [data-baseweb="notification"] {{
        background-color: #2D2D2D;
    }}
    
    /* Update markdown containers used in the app */
    div[data-testid="stMarkdownContainer"] {{
        color: #E0E0E0;
    }}
    </style>
    """, unsafe_allow_html=True)
def display_animated_stars(sentiment_score, show_number=True):
    if not isinstance(sentiment_score, (int, float)):
        return f"<span style='font-size: 18px;'>{sentiment_score}/5</span>"
    
    # Create animated stars HTML
    stars_html = ""
    for i in range(int(sentiment_score)):
        stars_html += f'<span class="animated-star" style="--star-index: {i}; font-size: 28px;">⭐</span>'
    
    # Add empty stars for the remaining
    for i in range(int(sentiment_score), 5):
        stars_html += f'<span style="font-size: 28px; color: #d0d0d0;">☆</span>'
    
    # Create the rating display with or without the number
    if show_number:
        rating_html = f'<div style="font-size: 24px;">{stars_html} <span style="font-size: 18px; color: #666; margin-left: 10px;">({sentiment_score}/5)</span></div>'
    else:
        rating_html = f'<div style="font-size: 24px;">{stars_html}</div>'
    
    return rating_html

# Helper functions for displaying reviews
def format_date(timestamp, format_str="%b %d, %Y at %I:%M %p"):
    if timestamp == 'Unknown date':
        return timestamp
    try:
        return datetime.fromisoformat(timestamp).strftime(format_str)
    except:
        return timestamp[0:10]  # Extract just the date part

def display_list_items(items):
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                clean_item = item.strip().strip("'").strip('"')
                st.markdown(f"• {clean_item}")
    elif isinstance(items, str):
        # Handle string representation of list
        if items.startswith('[') and items.endswith(']'):
            try:
                # Try to convert to list
                import ast
                items_list = ast.literal_eval(items)
                for item in items_list:
                    clean_item = str(item).strip().strip("'").strip('"')
                    st.markdown(f"• {clean_item}")
            except:
                # Handle as comma-separated string
                item_parts = items.strip("[]").split(',')
                for item in item_parts:
                    clean_item = item.strip().strip("'").strip('"')
                    st.markdown(f"• {clean_item}")
        else:
            # Handle as comma-separated string
            item_parts = items.split(',')
            for item in item_parts:
                clean_item = item.strip().strip("'").strip('"')
                st.markdown(f"• {clean_item}")
def ensure_directories():
    """Ensure necessary directories exist for file storage"""
    # Create a directory for audio recordings if it doesn't exist
    audio_dir = "audio_recordings"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    return audio_dir

def generate_safe_timestamp():
    """Generate a timestamp that's safe for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
# Main application
def main():
    try:
        # Initialize state and styles
        init_session_state()
        load_css()
        
        # Ensure necessary directories exist
        ensure_directories()
        
        # App header with logo and title
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("https://emojipedia-us.s3.amazonaws.com/source/microsoft-teams/337/fork-and-knife-with-plate_1f37d-fe0f.png", width=80)
        with col2:
            st.title("Restaurant Feedback Portal")
            st.markdown("<p style='font-size: 18px; margin-top: -10px; color: #E0E0E0;'>Share your dining experience and help us improve!</p>", unsafe_allow_html=True)
        
        # Sidebar with improved styling
        with st.sidebar:
            st.header("About this system")
            st.markdown(f"""
            <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                 box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <p style="margin-top: 0; color: #E0E0E0;">Share your dining experience with us! Your feedback helps us improve.</p>
                <p style="margin-bottom: 10px; color: #E0E0E0;">This system will:</p>
                <ol style="padding-left: 20px; margin-bottom: 0; color: #E0E0E0;">
                    <li style="margin-bottom: 8px;">Record your verbal feedback (25 seconds)</li>
                    <li style="margin-bottom: 8px;">Transcribe what you said</li>
                    <li style="margin-bottom: 8px;">Analyze your feedback</li>
                    <li style="margin-bottom: 0;">Save your insights for the restaurant</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # User account section with improved styling
            st.header("Your Account")
            
            # If user is logged in, show user info and logout button
            if check_login_status():
                st.markdown(f"""
                <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                     box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 50px; height: 50px; border-radius: 50%; background-color: {BRAND_COLOR}; 
                             color: white; display: flex; align-items: center; justify-content: center; 
                             font-size: 20px; margin-right: 15px;">
                            {st.session_state.current_user.get('name', 'U')[0].upper()}
                        </div>
                        <div>
                            <p style="margin: 0; font-weight: 600; font-size: 18px; color: #FFFFFF;">
                                {st.session_state.current_user.get('name', 'User')}
                            </p>
                            <p style="margin: 0; color: #BBBBBB; font-size: 14px;">
                                {st.session_state.current_user.get('email', 'N/A')}
                            </p>
                        </div>
                    </div>
                    <p style="margin: 0; font-size: 14px; color: #BBBBBB;">
                        Last login: {format_date(st.session_state.current_user.get('last_login', 'Unknown'))}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚪 Logout", use_container_width=True):
                    logout()
            
            # Show recent reviews with improved styling
            if st.session_state.reviews:
                st.header("Recent Feedback")
                recent_reviews = sorted(
                    st.session_state.reviews, 
                    key=lambda x: x.get('timestamp', ''), 
                    reverse=True
                )[:3]
                
                for i, review in enumerate(recent_reviews):
                    display_date = format_date(review.get('timestamp', 'Unknown date'), "%b %d, %Y")
                    
                    with st.expander(f"Review {i+1} - {display_date}"):
                        # Sentiment display with animated stars
                        sentiment = review.get('sentiment_score', 'N/A')
                        if isinstance(sentiment, (int, float)):
                            animated_stars = display_animated_stars(sentiment, show_number=False)
                            st.markdown(animated_stars, unsafe_allow_html=True)
                        else:
                            st.write(f"**Sentiment**: {sentiment}/5")
                            
                        st.write(f"**Summary**: {review.get('summary', 'N/A')}")
        
        # Check if user is logged in before showing main content
        if not check_login_status():
            # User is not logged in, show login form
            render_login_form()
            
            if st.session_state.register_success:
                st.success(st.session_state.register_success)
        else:
            # User is logged in, show main content with improved tab styling
            if st.session_state.is_owner:
                # Show all tabs for owner
                tab1, tab2, tab3 = st.tabs(["📝 Leave Feedback", "📋 View All Feedback", "👤 My Feedback"])
            else:
                # Show only Leave Feedback and My Feedback tabs for regular users
                tab1, tab3 = st.tabs(["📝 Leave Feedback", "👤 My Feedback"])

            # Feedback tab
            with tab1:
                st.markdown(f"""
                <h2 style="color: #FFFFFF; margin-bottom: 20px;">
                    <span style="margin-right: 10px;">📝</span> Share Your Experience
                </h2>
                <p style="font-size: 16px; margin-bottom: 20px; color: #E0E0E0;">
                    Please tell us about your dining experience at our restaurant. Your feedback helps us improve!
                </p>
                """, unsafe_allow_html=True)
                
                # Collect customer information
                collect_customer_info()
                    
                col1, col2 = st.columns(2)
                
                # Left column - Audio recording with improved styling
                with col1:
                    st.markdown(f"""
                    <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                         box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;">
                        <h3 style="color: #FFFFFF; margin-top: 0; margin-bottom: 15px; font-size: 20px;">
                            <span style="margin-right: 8px;">🎙️</span> Voice Feedback
                        </h3>
                        <p style="margin-bottom: 20px; color: #E0E0E0;">
                            Record your feedback by speaking into your microphone. This is the fastest way to share your experience.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show recording UI if not showing analysis
                    if not st.session_state.show_analysis:
                        audio_file = record_audio()
                    
                    with st.expander("Having Audio Problems?"):
                        st.markdown("""
                        ### Microphone Access
                        - **Mobile**: Ensure your browser has microphone permissions
                        - **Desktop**: Check your system's sound settings and browser permissions
                        - **iOS devices**: Use Safari for best microphone support
                        - **Android**: Make sure microphone access is enabled for your browser
                        
                        ### Audio Quality Issues
                        - Try moving to a quieter environment
                        - Speak a bit louder than your normal speaking voice
                        - Make sure you're not covering the microphone
                        - On mobile, don't cover the bottom of your device
                        
                        ### If Recording Doesn't Work
                        - Refresh the page and try again
                        - Try a different browser (Chrome or Safari recommended)
                        - Consider using the text input method instead
                        """)

                # Display analysis and save options
                if st.session_state.show_analysis and st.session_state.current_analysis:
                    with col1:
                        display_analysis(st.session_state.current_analysis)
                        
                        # Save/Cancel buttons with improved styling
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("💾 Save Feedback", type="primary", use_container_width=True):
                                if save_review(st.session_state.current_analysis):
                                    st.success("Thank you! Your feedback has been saved.")
                                    # Reset states
                                    st.session_state.audio_file = None
                                    st.session_state.show_analysis = False
                                    st.session_state.current_analysis = None
                                    st.session_state.is_recording = False
                                    st.rerun()
                        
                        with col_cancel:
                            if st.button("↩️ Start Over", use_container_width=True):
                                # Reset states with error handling
                                if hasattr(st.session_state, 'audio_file') and st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                                    try:
                                        os.remove(st.session_state.audio_file)
                                    except Exception as e:
                                        print(f"Error removing audio file: {str(e)}")
                                        
                                st.session_state.audio_file = None
                                st.session_state.show_analysis = False
                                st.session_state.current_analysis = None
                                st.session_state.is_recording = False
                                st.session_state.record_again = True
                                st.rerun()

                # Text input - Right column with improved styling
                with col2:
                    if not st.session_state.show_analysis:
                        st.markdown(f"""
                        <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                             box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;">
                            <h3 style="color: #FFFFFF; margin-top: 0; margin-bottom: 15px; font-size: 20px;">
                                <span style="margin-right: 8px;">✏️</span> Written Feedback
                            </h3>
                            <p style="margin-bottom: 20px; color: #E0E0E0;">
                                Prefer typing? Share your experience by writing your feedback below.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        text_feedback = st.text_area("Your feedback", height=200, 
                                                   placeholder="Tell us about your experience... What did you enjoy? What could we improve?")
                        
                        if st.button("📝 Submit Written Feedback", type="primary", use_container_width=True):
                            if text_feedback:
                                with st.spinner("Analyzing your feedback..."):
                                    review_analysis, validation_error = process_and_validate_review(text_feedback)
                                
                                if validation_error:
                                    st.error(validation_error)
                                    st.info("Please provide more details about your restaurant experience.")
                                elif review_analysis:
                                    st.success("Analysis complete!")
                                    st.session_state.current_analysis = review_analysis
                                    st.session_state.show_analysis = True
                                    st.rerun()
                            else:
                                st.warning("Please enter your feedback before submitting.")

            # View all feedback tab with improved styling
            if st.session_state.is_owner:
                with tab2:
                    st.markdown(f"""
                    <h2 style="color: #FFFFFF; margin-bottom: 20px;">
                        <span style="margin-right: 10px;">📋</span> All Customer Feedback
                    </h2>
                    <p style="font-size: 16px; margin-bottom: 20px; color: #E0E0E0;">
                        Review all customer feedback to identify trends and improvement opportunities.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    # Get total count first for reference
                    all_reviews_total = load_reviews_from_db()
                    total_reviews_count = len(all_reviews_total)
                    
                    # Load only the latest 20 reviews for display
                    all_reviews = load_reviews_from_db(limit=20)
                    
                    if not all_reviews:
                        st.info("No feedback has been submitted yet.")
                    else:
                        # Sort reviews by timestamp
                        sorted_reviews = sorted(all_reviews, key=lambda x: x.get('timestamp', ''), reverse=True)
                        
                        # Filter controls with improved styling
                        st.markdown(f"""
                        <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                             box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px;">
                            <h3 style="color: #FFFFFF; margin-top: 0; margin-bottom: 15px; font-size: 18px;">
                                <span style="margin-right: 8px;">🔍</span> Filter Reviews
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        search_input = st.text_input("Search reviews by keyword:", placeholder="Type to search...")
                        
                        # Sentiment filter
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment_options = ["All"] + [str(i) for i in range(1, 6)]
                            selected_sentiment = st.selectbox("Filter by sentiment score:", sentiment_options)
                        
                        # Date filter
                        col_date1, col_date2 = st.columns(2)
                        with col_date1:
                            # Get min date
                            dates = [datetime.fromisoformat(r.get('timestamp', datetime.now().isoformat())) 
                                    for r in all_reviews if 'timestamp' in r]
                            min_date = min(dates).date() if dates else datetime.now().date()
                            start_date = st.date_input("From date:", min_date)
                        
                        with col_date2:
                            max_date = max(dates).date() if dates else datetime.now().date()
                            end_date = st.date_input("To date:", max_date)
                        
                        # Apply filters
                        filtered_reviews = []
                        for review in sorted_reviews:
                            # Filter by sentiment
                            if selected_sentiment != "All":
                                sentiment_score = review.get('sentiment_score', None)
                                if sentiment_score != float(selected_sentiment) and sentiment_score != int(selected_sentiment):
                                    continue
                            
                            # Filter by date
                            if 'timestamp' in review:
                                try:
                                    review_date = datetime.fromisoformat(review['timestamp']).date()
                                    if review_date < start_date or review_date > end_date:
                                        continue
                                except:
                                    pass
                            
                            # Add to filtered list
                            filtered_reviews.append(review)
                        
                        # Display count with message about limit
                        st.markdown(f"""
                        <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin: 20px 0;">
                            <p style="margin: 0; font-size: 16px; color: #E0E0E0;">
                        """, unsafe_allow_html=True)
                        
                        if total_reviews_count > 20:
                            st.markdown(f"Showing up to 20 most recent reviews (out of {total_reviews_count} total)")
                        else:
                            st.markdown(f"Showing {len(filtered_reviews)} out of {total_reviews_count} total reviews")
                        
                        st.markdown("</p></div>", unsafe_allow_html=True)
                        
                        # Display filtered reviews with improved styling
                        for review in filtered_reviews:
                            display_date = format_date(review.get('timestamp', 'Unknown date'))

                            with st.expander(f"Review from {display_date}"):
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                                     box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px;">
                                    <div style="display: flex; flex-wrap: wrap; margin-bottom: 15px; 
                                         padding-bottom: 15px; border-bottom: 1px solid #444444;">
                                        <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                                            <p style="margin: 0; color: #BBBBBB; font-size: 14px;">Customer</p>
                                            <p style="margin: 0; font-weight: 500; font-size: 16px; color: #FFFFFF;">
                                                {review.get('customer_name', 'Anonymous')}
                                            </p>
                                        </div>
                                        <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                                            <p style="margin: 0; color: #BBBBBB; font-size: 14px;">Email</p>
                                            <p style="margin: 0; font-weight: 500; font-size: 16px; color: #FFFFFF;">
                                                {review.get('customer_email', 'N/A')}
                                            </p>
                                        </div>
                                        <div style="flex: 1; min-width: 200px; margin-bottom: 10px;">
                                            <p style="margin: 0; color: #BBBBBB; font-size: 14px;">Phone</p>
                                            <p style="margin: 0; font-weight: 500; font-size: 16px; color: #FFFFFF;">
                                                {review.get('customer_phone', 'Not provided')}
                                            </p>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Review details
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Summary
                                        </h4>
                                        <p style="margin: 0; color: #E0E0E0;">{review.get('summary', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Sentiment display with animated stars
                                    sentiment = review.get('sentiment_score', 'N/A')
                                    if isinstance(sentiment, (int, float)):
                                        animated_stars = display_animated_stars(sentiment, show_number=True)
                                        st.markdown(f"""
                                        <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center;">
                                            <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                                Sentiment Score
                                            </h4>
                                            {animated_stars}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.write(f"**Sentiment Score**: {sentiment}/5")
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Food Quality
                                        </h4>
                                        <p style="margin: 0; color: #E0E0E0;">{review.get('food_quality', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Service
                                        </h4>
                                        <p style="margin: 0; color: #E0E0E0;">{review.get('service', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                col3, col4 = st.columns(2)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Atmosphere
                                        </h4>
                                        <p style="margin: 0; color: #E0E0E0;">{review.get('atmosphere', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Music & Entertainment
                                        </h4>
                                        <p style="margin: 0; color: #E0E0E0;">{review.get('music_and_entertainment', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Display key points
                                if 'specific_points' in review:
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Key Points
                                        </h4>
                                        <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
                                    """, unsafe_allow_html=True)
                                    
                                    display_list_items(review.get('specific_points', []))
                                    
                                    st.markdown("</ul></div>", unsafe_allow_html=True)

                                # Display suggestions
                                if 'improvement_suggestions' in review:
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Suggestions for Improvement
                                        </h4>
                                        <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
                                    """, unsafe_allow_html=True)
                                    
                                    display_list_items(review.get('improvement_suggestions', []))
                                    
                                    st.markdown("</ul></div>", unsafe_allow_html=True)
                                    
                                # Show transcription
                                if 'raw_transcription' in review:
                                    with st.expander("View Original Transcription"):
                                        st.text(review['raw_transcription'])
                                
                                st.markdown("</div>", unsafe_allow_html=True)
            
            # My Feedback tab with improved styling
            with tab3:
                st.markdown(f"""
                <h2 style="color: #FFFFFF; margin-bottom: 20px;">
                    <span style="margin-right: 10px;">👤</span> My Feedback History
                </h2>
                <p style="font-size: 16px; margin-bottom: 20px; color: #E0E0E0;">
                    View all the feedback you've submitted to our restaurant.
                </p>
                """, unsafe_allow_html=True)
                
                # Get user ID
                user_id = st.session_state.customer_id
                
                # Get total user reviews count for reference
                all_user_reviews = get_user_reviews(user_id)
                total_user_reviews_count = len(all_user_reviews)
                
                # Get limited user reviews (10 most recent)
                user_reviews = get_user_reviews(user_id, limit=10)
                
                if not user_reviews:
                    st.info("You haven't submitted any feedback yet.")
                else:
                    # Display count with message about limit
                    st.markdown(f"""
                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <p style="margin: 0; font-size: 16px; color: #E0E0E0;">
                    """, unsafe_allow_html=True)
                    
                    if total_user_reviews_count > 10:
                        st.markdown(f"Showing your 10 most recent reviews (out of {total_user_reviews_count} total)")
                    else:
                        st.markdown(f"You have submitted {total_user_reviews_count} reviews.")
                    
                    st.markdown("</p></div>", unsafe_allow_html=True)
                    
                    # Display user reviews with improved styling
                    for i, review in enumerate(user_reviews):
                        display_date = format_date(review.get('timestamp', 'Unknown date'))

                        with st.expander(f"Review {i+1} - {display_date}"):
                            st.markdown(f"""
                            <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                                 box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px;">
                            """, unsafe_allow_html=True)
                            
                            # Review details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Summary
                                    </h4>
                                    <p style="margin: 0; color: #E0E0E0;">{review.get('summary', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Sentiment display with animated stars
                                sentiment = review.get('sentiment_score', 'N/A')
                                if isinstance(sentiment, (int, float)):
                                    animated_stars = display_animated_stars(sentiment, show_number=True)
                                    st.markdown(f"""
                                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center;">
                                        <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                            Sentiment Score
                                        </h4>
                                        {animated_stars}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.write(f"**Sentiment Score**: {sentiment}/5")
                            
                            with col2:
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Food Quality
                                    </h4>
                                    <p style="margin: 0; color: #E0E0E0;">{review.get('food_quality', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Service
                                    </h4>
                                    <p style="margin: 0; color: #E0E0E0;">{review.get('service', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Atmosphere
                                    </h4>
                                    <p style="margin: 0; color: #E0E0E0;">{review.get('atmosphere', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: {BRAND_COLOR}; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Music & Entertainment
                                    </h4>
                                    <p style="margin: 0;">{review.get('music_and_entertainment', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                            # Display key points
                            if 'specific_points' in review:
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Key Points
                                    </h4>
                                    <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
                                """, unsafe_allow_html=True)
                                
                                display_list_items(review.get('specific_points', []))
                                
                                st.markdown("</ul></div>", unsafe_allow_html=True)

                            # Display suggestions
                            if 'improvement_suggestions' in review:
                                st.markdown(f"""
                                <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 16px;">
                                        Suggestions for Improvement
                                    </h4>
                                    <ul style="margin-bottom: 0; padding-left: 20px; color: #E0E0E0;">
                                """, unsafe_allow_html=True)
                                
                                display_list_items(review.get('improvement_suggestions', []))
                                
                                st.markdown("</ul></div>", unsafe_allow_html=True)
                                
                            # Show transcription
                            if 'raw_transcription' in review:
                                with st.expander("View Original Transcription"):
                                    st.text(review['raw_transcription'])
                            
                            st.markdown("</div>", unsafe_allow_html=True)

        # Footer with improved styling
        st.markdown(f"""
        <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #444444;">
            <p style="color: #BBBBBB; font-size: 14px;">
                Thank you for sharing your feedback! • © {datetime.now().year} Restaurant Feedback System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
