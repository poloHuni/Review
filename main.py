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
import streamlit_nested_layout

# Google OAuth imports
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pathlib

# Constants
OWNER_EMAIL = "natnaelgebremichaeltewelde@gmail.com"
BRAND_COLOR = "#f8f9fa"
ACCENT_COLOR = "#e53935"
LIGHT_COLOR = "#5a7d7c"

# Google OAuth config
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8501")
GOOGLE_SCOPES = [
    'openid',  # Add this scope to match what Google automatically includes
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/business.manage'  # For Google Business Profile (Reviews)
]

load_dotenv()

# Setup Google OAuth flow
def create_google_oauth_flow():
    """Create a new OAuth flow instance"""
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI]
        }
    }
    
    # Verify client credentials are available
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        st.error("Google OAuth credentials are missing. Check your environment variables.")
        st.stop()
    
    # Create flow with all required parameters
    flow = Flow.from_client_config(
        client_config=client_config,
        scopes=GOOGLE_SCOPES
    )
    
    # Set the redirect_uri explicitly
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    
    return flow

def get_google_oauth_url():
    """Generate the authorization URL for Google OAuth"""
    flow = create_google_oauth_flow()
    
    # Generate a random state parameter for additional security
    import secrets
    state = secrets.token_urlsafe(16)
    st.session_state.oauth_state = state
    
    # Explicitly include the openid scope in both authorization and token steps
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        state=state
    )
    
    # Save the scopes that were used in the authorization request
    st.session_state.oauth_scopes = flow.oauth2session.scope
    
    return auth_url

def process_oauth_callback(code):
    """Process the OAuth callback code and get user info"""
    try:
        flow = create_google_oauth_flow()
        flow.redirect_uri = GOOGLE_REDIRECT_URI
        
        st.write(f"Debug - Attempting token exchange with code length: {len(code)}")
        
        # Adding parameters to handle scope mismatch
        token_info = flow.fetch_token(
            code=code,
            # Include these parameters
            include_client_id=True,
            # Tell the library not to verify scopes, which allows for scope differences
            verify_scope=False
        )
        
        st.write("Debug - Token exchange successful")
        st.write(f"Debug - Access token received (length: {len(token_info.get('access_token', ''))})")
        
        credentials = flow.credentials
        
        # Get user info
        user_info = get_google_user_info(credentials)
        
        if user_info:
            st.write(f"Debug - User info retrieved: {user_info.get('email')}")
            
            # Create or update user in database
            user_id = str(uuid.uuid4())
            existing_user = find_user_by_email(user_info.get('email', ''))
            
            if existing_user:
                user_id = existing_user.get('user_id')
                st.write(f"Debug - Existing user found with ID: {user_id}")
            else:
                st.write(f"Debug - Creating new user with ID: {user_id}")
            
            user_data = {
                "user_id": user_id,
                "username": user_info.get('email', '').split('@')[0],
                "email": user_info.get('email', ''),
                "phone": existing_user.get('phone', '') if existing_user else '',
                "name": user_info.get('name', ''),
                "last_login": datetime.now().isoformat(),
                "registration_date": existing_user.get('registration_date', datetime.now().isoformat()) if existing_user else datetime.now().isoformat(),
                "google_id": user_info.get('id', ''),
                "picture": user_info.get('picture', ''),
                "credentials": credentials_to_dict(credentials)
            }
            
            if save_user(user_data):
                st.write("Debug - User saved successfully")
                
                # Set cookie
                cookie_manager = get_cookie_manager()
                cookie_manager.set(
                    cookie="user_id", 
                    val=user_id,
                    expires_at=datetime.now() + timedelta(days=30)
                )
                
                # Update session state
                st.session_state.is_logged_in = True
                st.session_state.current_user = user_data
                st.session_state.customer_id = user_id
                st.session_state.customer_info = {
                    "name": user_info.get('name', ''),
                    "email": user_info.get('email', ''),
                    "phone": existing_user.get('phone', '') if existing_user else ''
                }
                
                # Check if this user is the owner
                st.session_state.is_owner = (user_info.get('email', '').lower() == OWNER_EMAIL.lower())
                
                return True
            else:
                st.error("Failed to save user data")
        else:
            st.error("Failed to get user info")
        
        return False
    except Exception as e:
        st.error(f"OAuth error: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        
        # More specific error handling
        if "invalid_grant" in str(e).lower():
            st.error("The authorization code has expired or already been used. Please try logging in again.")
        elif "redirect_uri_mismatch" in str(e).lower():
            st.error(f"Redirect URI mismatch. Expected: {GOOGLE_REDIRECT_URI}")
        elif "scope" in str(e).lower():
            st.error("OAuth scope mismatch. This is likely due to Google automatically adding scopes.")
        
        import traceback
        st.error(traceback.format_exc())
        return False
        # Get Google user info
def get_google_user_info(credentials):
    try:
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        return user_info
    except HttpError as e:
        st.error(f"Error getting user info: {str(e)}")
        return None

# Convert credentials to dict for storage
def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

# Convert dict to credentials
def dict_to_credentials(credentials_dict):
    return Credentials(
        token=credentials_dict.get('token', ''),
        refresh_token=credentials_dict.get('refresh_token', ''),
        token_uri=credentials_dict.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=credentials_dict.get('client_id', GOOGLE_CLIENT_ID),
        client_secret=credentials_dict.get('client_secret', GOOGLE_CLIENT_SECRET),
        scopes=credentials_dict.get('scopes', GOOGLE_SCOPES)
    )
# Post review to Google
def post_to_google_reviews(review_data, user):
    try:
        if not user or 'credentials' not in user:
            return False, "User credentials not found"
            
        credentials = dict_to_credentials(user['credentials'])
        
        # This is a simplified example - actual implementation requires Google My Business API
        # which has specific requirements and limitations
        # Usually this would be done through the Google My Business API
        
        # For now, we'll just log that we would post to Google
        st.success("Your review would be posted to Google Reviews!")
        
        # In a real implementation, you would use the Google My Business API
        # to post the review to the business's Google profile
        
        return True, "Review posted successfully"
    except Exception as e:
        return False, f"Error posting to Google: {str(e)}"


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
    # Use Streamlit's native components instead of custom HTML/CSS
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <span style="font-size: 32px;">👋</span>
        <h1 style="color: #FFFFFF; margin-top: 10px;">Welcome to my Feedback Portal</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a card-like container for the login form using Streamlit components
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Login section
            st.subheader("Login")
            st.markdown("<p style='color: #CCCCCC;'>Access your account to share your feedback and see your previous reviews.</p>", 
                       unsafe_allow_html=True)
            
            # Google Sign-In button
            google_auth_url = get_google_oauth_url()
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <a href="{google_auth_url}" target="_self" style="text-decoration: none;">
                    <div style="background-color: white; color: #444; padding: 10px 20px; border-radius: 4px; 
                         display: inline-flex; align-items: center; font-family: 'Roboto', sans-serif; 
                         font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.25);">
                        <img src="https://developers.google.com/identity/images/g-logo.png" 
                             style="height: 24px; margin-right: 10px;">
                        Sign in with Google
                    </div>
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='text-align: center; color: #999; margin: 15px 0;'>- or -</p>", unsafe_allow_html=True)
            
            with st.form("login_form", border=False):
                email = st.text_input("Email Address", key="login_email")
                submit_login = st.form_submit_button("Login with Email", use_container_width=True)
                
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
    
    # Registration section - completely redesigned to avoid any class issues
    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.subheader("New User? Register Here")
        st.markdown("<p style='color: #CCCCCC;'>Create an account to start sharing your dining experiences with me.</p>", 
                   unsafe_allow_html=True)
        
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
        return False, "Your feedback seems too short. Please share more details about your experience."
    
    initialize_conversation()
    
    validation_prompt = f"""
    You need to determine if the following text is valid feedback about a restaurant or not.
    
    Text: "{text}"
    
    Valid feedback typically mentions food, service, atmosphere, or specific experiences at a restaurant.
    Greetings, questions about the system, or unrelated conversations are NOT valid feedback.
    
    Return a JSON object with this format:
    {{
        "is_valid": true/false,
        "reason": "Brief explanation of why it is or isn't valid feedback",
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
        return False, "I couldn't validate your feedback. Please make sure you're sharing thoughts about your restaurant experience."
    
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

def generate_google_review_link(review_data, place_id):
    """
    Generate a URL that directs users to Google Reviews with pre-filled content.
    
    Args:
        review_data: The processed review data dictionary
        place_id: Your Google Business Profile Place ID
    
    Returns:
        str: URL to Google Review page with pre-filled content
    """
    # Extract key components from the review
    summary = review_data.get('summary', '')
    food_quality = review_data.get('food_quality', '')
    service = review_data.get('service', '')
    atmosphere = review_data.get('atmosphere', '')
    
    # Create a formatted review text
    review_text = f"{summary}\n\n"
    
    if food_quality and food_quality != "N/A":
        review_text += f"Food: {food_quality}\n"
    if service and service != "N/A":
        review_text += f"Service: {service}\n"
    if atmosphere and atmosphere != "N/A":
        review_text += f"Atmosphere: {atmosphere}\n"
    
    # Add specific points if available
    if 'specific_points' in review_data and review_data['specific_points']:
        review_text += "\nHighlights:\n"
        if isinstance(review_data['specific_points'], list):
            for point in review_data['specific_points']:
                if point and point != "N/A":
                    review_text += f"- {point}\n"
        elif isinstance(review_data['specific_points'], str):
            # Handle string representation of list
            points = review_data['specific_points'].strip("[]").split(',')
            for point in points:
                clean_point = point.strip().strip("'").strip('"')
                if clean_point and clean_point != "N/A":
                    review_text += f"- {clean_point}\n"
    
    # URL encode the review text
    import urllib.parse
    encoded_review = urllib.parse.quote(review_text)
    
    # Create the Google review URL with the place ID and pre-filled review
    # Note: The 'review' URL parameter isn't officially documented by Google, but works in many cases
    review_url = f"https://search.google.com/local/writereview?placeid={place_id}&review={encoded_review}"
    
    return review_url

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
                text="Click to begin/end recording",
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
    
    # Updated prompt with more explicit JSON formatting instructions
    prompt = f"""
    I want you to analyze my feedback for this bar and restaurant which features live DJs and music.
    The feedback was: "{transcribed_text}"
    
    Format this feedback as a first-person review that I can copy and paste directly to Google Reviews.
    All assessments, opinions, and points should be written in first-person (using "I", "my", "me").
    
    For example, instead of "The customer enjoyed the food" write "I enjoyed the food".
    Instead of "The customer thought the music was too loud" write "I thought the music was too loud".
    
    Provide your analysis in the following JSON format:
    {{
        "summary": "A brief first-person summary of my overall experience",
        "food_quality": "My assessment of food and drinks",
        "service": "My assessment of service quality",
        "atmosphere": "My assessment of ambiance, music, and entertainment",
        "music_and_entertainment": "My specific feedback on DJs, music selection, and overall vibe",
        "specific_points": ["My point 1", "My point 2", "My point 3"],
        "sentiment_score": 4,
        "improvement_suggestions": ["My suggestion 1", "My suggestion 2"]
    }}
    
    MAKE SURE that:
    1. All text fields are properly enclosed in double quotes
    2. All arrays have square brackets and comma-separated values in double quotes
    3. The sentiment_score is a number between 1-5 without quotes
    4. There is no trailing comma after the last item in arrays or objects
    5. Your response contains ONLY the JSON object - no other text before or after
    """
    
    response = st.session_state.conversation.predict(input=prompt)
    
    try:
        # First try: direct JSON parsing
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            # Second try: Extract JSON using regex
            json_pattern = r'(\{[\s\S]*\})'
            match = re.search(json_pattern, response)
            
            if match:
                json_str = match.group(1)
                # Try to fix common JSON errors
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)  # Add quotes to keys
                # Remove trailing commas before closing brackets/braces (common JSON error)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                parsed_response = json.loads(json_str)
            else:
                # If regex fails, create a manually constructed response
                raise ValueError("Could not extract valid JSON from response")
            
        # Add raw transcription
        parsed_response["raw_transcription"] = transcribed_text
        
        # Validate required fields and provide defaults if missing
        required_fields = [
            "summary", "food_quality", "service", "atmosphere", 
            "music_and_entertainment", "specific_points", 
            "sentiment_score", "improvement_suggestions"
        ]
        
        for field in required_fields:
            if field not in parsed_response:
                # Provide default values based on field type
                if field == "sentiment_score":
                    parsed_response[field] = 3  # Neutral default
                elif field in ["specific_points", "improvement_suggestions"]:
                    # Extract relevant points from the text if possible
                    parts = transcribed_text.split('.')
                    points = [p.strip() for p in parts if len(p.strip()) > 15][:3]  # Use up to 3 sentences as points
                    if not points:
                        points = ["Based on my experience"]
                    
                    # Convert to first person if not already
                    first_person_points = []
                    for point in points:
                        if not any(fp in point.lower() for fp in ["i ", "my ", "me ", "we ", "our "]):
                            # Convert to first person if possible
                            point = "I " + point[0].lower() + point[1:]
                        first_person_points.append(point)
                    
                    parsed_response[field] = first_person_points
                else:
                    # For text fields, provide a meaningful default based on the raw transcription
                    if field == "summary":
                        # Use first sentence of transcription as summary
                        first_sentence = transcribed_text.split('.')[0] + "."
                        parsed_response[field] = f"I visited this place and {first_sentence[0].lower() + first_sentence[1:]}"
                    else:
                        parsed_response[field] = f"Based on my experience, the {field.replace('_', ' ')} was satisfactory."
        
        return parsed_response
        
    except Exception as e:
        # Gracefully handle any errors with useful defaults
        st.warning(f"There was an issue processing your review, but we've created a basic summary for you.")
        
        # Create a basic but useful response from the raw transcription
        # Parse the raw text to extract meaningful content for the review
        sentences = [s.strip() for s in transcribed_text.split('.') if len(s.strip()) > 0]
        
        # Convert sentences to first person if they aren't already
        first_person_sentences = []
        for sentence in sentences:
            if not any(fp in sentence.lower() for fp in ["i ", "my ", "me ", "we ", "our "]):
                # Convert to first person
                sentence = "I " + sentence[0].lower() + sentence[1:]
            first_person_sentences.append(sentence)
        
        # Create reasonable defaults for each field
        summary = first_person_sentences[0] if sentences else "I visited this restaurant."
        
        # Extract specific points based on sentence length and content
        points = []
        suggestions = []
        
        for s in first_person_sentences[1:]:
            if "should" in s.lower() or "could" in s.lower() or "wish" in s.lower() or "hope" in s.lower():
                suggestions.append(s)
            elif len(s) > 15:  # Only use substantial sentences
                points.append(s)
        
        # Ensure we have at least some points and suggestions
        if not points and len(first_person_sentences) > 1:
            points = [first_person_sentences[1]]
        
        if not suggestions and points:
            suggestions = ["I hope they continue providing this quality of experience."]
        
        return {
            "summary": summary,
            "food_quality": "I enjoyed the food based on my visit." if "food" in transcribed_text.lower() else "N/A",
            "service": "The service was good during my visit." if "service" in transcribed_text.lower() else "N/A",
            "atmosphere": "I liked the atmosphere of the place." if "atmosphere" in transcribed_text.lower() else "N/A",
            "music_and_entertainment": "The music added to my experience." if any(x in transcribed_text.lower() for x in ["music", "dj", "entertainment"]) else "N/A",
            "specific_points": points[:3] if points else ["I had an experience worth sharing."],
            "sentiment_score": 4 if any(p in transcribed_text.lower() for p in ["good", "great", "excellent", "amazing", "enjoyed"]) else 3,
            "improvement_suggestions": suggestions[:2] if suggestions else ["Based on my experience, I think they're doing well."],
            "raw_transcription": transcribed_text
        }
    
# Here's the updated display_analysis function with simplified buttons

# Here's the updated display_analysis function with simplified buttons

def display_analysis(review_analysis):
    with st.container():
        st.markdown(f"""
        <div style="padding: 25px; border-radius: 10px; border: 1px solid #444444; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); background-color: #2D2D2D; margin-bottom: 20px;">
            <h3 style="color: #FFFFFF; margin-top: 0; margin-bottom: 20px; font-size: 22px; 
                border-bottom: 2px solid {BRAND_COLOR}; padding-bottom: 10px;">
                Your Feedback Analysis
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
        
        # Format the review for sharing and copying
        formatted_review = format_review_for_sharing(review_analysis)
        
        # Create a container for the review text with copy functionality
        st.markdown(f"""
        <div style="margin-top: 20px; background-color: #333333; padding: 15px; border-radius: 8px;">
            <h4 style="color: #FFFFFF; margin-top: 0; margin-bottom: 10px; font-size: 18px;">
                <span style="margin-right: 8px;">📝</span> Your Review
            </h4>
            <p style="color: #E0E0E0; margin-bottom: 15px;">
                Here's your formatted review:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the review in a code block for easy copying
        st.code(formatted_review, language=None)
        
        # Simplified buttons - just Save Feedback and Start Over
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Feedback", 
                       use_container_width=True, 
                       type="primary",
                       key="save_feedback_display_btn"):
                if save_review(review_analysis):
                    st.success("Thanks for your feedback! It has been saved.")
                    st.balloons()
                    
                    # Get Google Place ID for direct link
                    PLACE_ID = os.environ.get("GOOGLE_PLACE_ID", "YOUR_PLACE_ID_HERE")
                    google_review_url = generate_google_review_link(review_analysis, PLACE_ID)
                    
                    # Display Google Review prompt with increased width
                    st.markdown(f"""
                    <div style="background-color: #4285F4; padding: 20px; border-radius: 10px; margin-top: 20px; 
                         box-shadow: 0 4px 12px rgba(0,0,0,0.2); width: 100%; max-width: 800px; margin-left: auto; margin-right: auto;">
                        <h4 style="color: #FFFFFF; margin-top: 0; font-size: 20px; margin-bottom: 15px; text-align: center;">
                            <span style="margin-right: 8px;">🌟</span> Share on Google Reviews?
                        </h4>
                        <p style="color: #FFFFFF; margin-bottom: 20px; text-align: center; font-size: 16px;">
                            Would you mind also sharing your experience on Google Reviews to help other diners?
                        </p>
                        <div style="display: flex; justify-content: center;">
                            <a href="{google_review_url}" target="_blank"
                               style="display: inline-block; background-color: #FFFFFF; color: #4285F4; 
                                      padding: 12px 25px; border-radius: 5px; text-decoration: none; 
                                      font-weight: 500; box-shadow: 0 2px 5px rgba(0,0,0,0.2); font-size: 16px;">
                                <img src="https://www.gstatic.com/images/branding/product/2x/googleg_48dp.png" 
                                     style="height: 20px; vertical-align: middle; margin-right: 8px;">
                                Open Google Reviews
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a "Done" button that works exactly like "Start Over"
                    if st.button("✅ Done", key="done_after_save", use_container_width=True):
                        # Reset states with error handling - exactly like "Start Over" button
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
                else:
                    st.error("Error saving your feedback. Please try again.")
        
        with col2:
            if st.button("↩️ Start Over", use_container_width=True, key="start_over_display_btn"):
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
        
        # End containers
        st.markdown("</div>", unsafe_allow_html=True)
        
        return False  # For compatibility with existing code

# Modified format_review_for_sharing function to create a clean, copyable format
def format_review_for_sharing(review_data):
    """
    Format review data for sharing as text.
    Returns a well-formatted string ready to be pasted into Google Reviews.
    """
    # Extract key components from the review
    summary = review_data.get('summary', '')
    food_quality = review_data.get('food_quality', '')
    service = review_data.get('service', '')
    atmosphere = review_data.get('atmosphere', '')
    music = review_data.get('music_and_entertainment', '')
    
    # Create a formatted review text
    review_text = f"{summary}\n\n"
    
    if food_quality and food_quality != "N/A":
        review_text += f"Food: {food_quality}\n"
    if service and service != "N/A":
        review_text += f"Service: {service}\n"
    if atmosphere and atmosphere != "N/A":
        review_text += f"Atmosphere: {atmosphere}\n"
    if music and music != "N/A":
        review_text += f"Music & Entertainment: {music}\n"
    
    # Add specific points if available
    specific_points = review_data.get('specific_points', [])
    if specific_points:
        review_text += "\nHighlights:\n"
        if isinstance(specific_points, list):
            for point in specific_points:
                if point and point != "N/A":
                    review_text += f"- {point}\n"
        elif isinstance(specific_points, str):
            # Handle string representation of list
            try:
                # Try to convert to list if it's a string representation of a list
                import ast
                points_list = ast.literal_eval(specific_points)
                for point in points_list:
                    clean_point = str(point).strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
            except:
                # If that fails, treat as comma-separated string
                points = specific_points.strip("[]").split(',')
                for point in points:
                    clean_point = point.strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
    
    return review_text

# Generate Google Review Link function (unchanged)
def generate_google_review_link(review_data, place_id):
    """
    Generate a URL that directs users to Google Reviews with pre-filled content.
    
    Args:
        review_data: The processed review data dictionary
        place_id: Your Google Business Profile Place ID
    
    Returns:
        str: URL to Google Review page with pre-filled content
    """
    # Extract key components from the review
    summary = review_data.get('summary', '')
    food_quality = review_data.get('food_quality', '')
    service = review_data.get('service', '')
    atmosphere = review_data.get('atmosphere', '')
    music = review_data.get('music_and_entertainment', '')
    
    # Create a formatted review text
    review_text = f"{summary}\n\n"
    
    if food_quality and food_quality != "N/A":
        review_text += f"Food: {food_quality}\n"
    if service and service != "N/A":
        review_text += f"Service: {service}\n"
    if atmosphere and atmosphere != "N/A":
        review_text += f"Atmosphere: {atmosphere}\n"
    if music and music != "N/A":
        review_text += f"Music & Entertainment: {music}\n"
    
    # Add specific points if available
    specific_points = review_data.get('specific_points', [])
    if specific_points:
        review_text += "\nHighlights:\n"
        if isinstance(specific_points, list):
            for point in specific_points:
                if point and point != "N/A":
                    review_text += f"- {point}\n"
        elif isinstance(specific_points, str):
            # Handle string representation of list
            try:
                # Try to convert to list if it's a string representation of a list
                import ast
                points_list = ast.literal_eval(specific_points)
                for point in points_list:
                    clean_point = str(point).strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
            except:
                # If that fails, treat as comma-separated string
                points = specific_points.strip("[]").split(',')
                for point in points:
                    clean_point = point.strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
    
    # URL encode the review text
    import urllib.parse
    encoded_review = urllib.parse.quote(review_text)
    
    # Create the Google review URL with the place ID and pre-filled review
    review_url = f"https://search.google.com/local/writereview?placeid={place_id}&review={encoded_review}"
    
    return review_url
                
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
def generate_google_review_link(review_data, place_id):
    """
    Generate a URL that directs users to Google Reviews with pre-filled content.
    
    Args:
        review_data: The processed review data dictionary
        place_id: Your Google Business Profile Place ID
    
    Returns:
        str: URL to Google Review page with pre-filled content
    """
    # Extract key components from the review
    summary = review_data.get('summary', '')
    food_quality = review_data.get('food_quality', '')
    service = review_data.get('service', '')
    atmosphere = review_data.get('atmosphere', '')
    music = review_data.get('music_and_entertainment', '')
    
    # Create a formatted review text
    review_text = f"{summary}\n\n"
    
    if food_quality and food_quality != "N/A":
        review_text += f"Food: {food_quality}\n"
    if service and service != "N/A":
        review_text += f"Service: {service}\n"
    if atmosphere and atmosphere != "N/A":
        review_text += f"Atmosphere: {atmosphere}\n"
    if music and music != "N/A":
        review_text += f"Music & Entertainment: {music}\n"
    
    # Add specific points if available
    specific_points = review_data.get('specific_points', [])
    if specific_points:
        review_text += "\nHighlights:\n"
        if isinstance(specific_points, list):
            for point in specific_points:
                if point and point != "N/A":
                    review_text += f"- {point}\n"
        elif isinstance(specific_points, str):
            # Handle string representation of list
            try:
                # Try to convert to list if it's a string representation of a list
                import ast
                points_list = ast.literal_eval(specific_points)
                for point in points_list:
                    clean_point = str(point).strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
            except:
                # If that fails, treat as comma-separated string
                points = specific_points.strip("[]").split(',')
                for point in points:
                    clean_point = point.strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
    
    # URL encode the review text
    import urllib.parse
    encoded_review = urllib.parse.quote(review_text)
    
    # Create the Google review URL with the place ID and pre-filled review
    review_url = f"https://search.google.com/local/writereview?placeid={place_id}&review={encoded_review}"
    
    return review_url

def format_review_for_sharing(review_data):
    """
    Format review data for sharing as text.
    Returns a well-formatted string ready to be pasted into Google Reviews.
    """
    # Extract key components from the review
    summary = review_data.get('summary', '')
    food_quality = review_data.get('food_quality', '')
    service = review_data.get('service', '')
    atmosphere = review_data.get('atmosphere', '')
    music = review_data.get('music_and_entertainment', '')
    
    # Create a formatted review text
    review_text = f"{summary}\n\n"
    
    if food_quality and food_quality != "N/A":
        review_text += f"Food: {food_quality}\n"
    if service and service != "N/A":
        review_text += f"Service: {service}\n"
    if atmosphere and atmosphere != "N/A":
        review_text += f"Atmosphere: {atmosphere}\n"
    if music and music != "N/A":
        review_text += f"Music & Entertainment: {music}\n"
    
    # Add specific points if available
    specific_points = review_data.get('specific_points', [])
    if specific_points:
        review_text += "\nHighlights:\n"
        if isinstance(specific_points, list):
            for point in specific_points:
                if point and point != "N/A":
                    review_text += f"- {point}\n"
        elif isinstance(specific_points, str):
            # Handle string representation of list
            try:
                # Try to convert to list if it's a string representation of a list
                import ast
                points_list = ast.literal_eval(specific_points)
                for point in points_list:
                    clean_point = str(point).strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
            except:
                # If that fails, treat as comma-separated string
                points = specific_points.strip("[]").split(',')
                for point in points:
                    clean_point = point.strip().strip("'").strip('"')
                    if clean_point and clean_point != "N/A":
                        review_text += f"- {clean_point}\n"
    
    return review_text
def create_google_review_modal(review_data, place_id):
    """
    Creates a modal popup to prompt the user to share their review on Google.
    
    Args:
        review_data: The review data dictionary
        place_id: Google Place ID for the business
    
    Returns:
        A Streamlit modal popup HTML string
    """
    # Generate the Google Review link with pre-filled content
    google_review_url = generate_google_review_link(review_data, place_id)
    
    # Format the review for copying
    formatted_review = format_review_for_sharing(review_data)
    
    # Create a styled modal popup - NOTE: Double curly braces for CSS in f-string
    modal_html = f"""
    <div id="google-review-modal" style="
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 1000;
        justify-content: center;
        align-items: center;
    ">
        <div style="
            background-color: #2D2D2D;
            width: 90%;
            max-width: 500px;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            padding: 25px;
            position: relative;
            animation: modalFadeIn 0.3s ease-out;
        ">
            <button onclick="closeModal()" style="
                position: absolute;
                top: 15px;
                right: 15px;
                background: none;
                border: none;
                font-size: 20px;
                color: #CCCCCC;
                cursor: pointer;
            ">×</button>
            
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="https://www.gstatic.com/images/branding/product/2x/googleg_48dp.png" 
                     style="height: 30px; margin-bottom: 15px;">
                <h3 style="color: #FFFFFF; margin: 0; font-size: 20px;">Share Your Review on Google</h3>
            </div>
            
            <p style="color: #E0E0E0; margin-bottom: 20px; text-align: center;">
                Thank you for your feedback! Would you like to help other diners by sharing your experience on Google Reviews?
            </p>
            
            <div style="
                background-color: #333333;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                max-height: 150px;
                overflow-y: auto;
            ">
                <p style="color: #BBBBBB; margin: 0 0 5px 0; font-size: 14px;">Your review:</p>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; white-space: pre-line;">{formatted_review}</p>
            </div>
            
            <div style="display: flex; justify-content: space-between;">
                <button onclick="closeModal()" style="
                    flex: 1;
                    background-color: #444444;
                    color: #FFFFFF;
                    border: none;
                    padding: 12px;
                    border-radius: 6px;
                    margin-right: 10px;
                    cursor: pointer;
                    font-weight: 500;
                ">Maybe Later</button>
                
                <a href="{google_review_url}" target="_blank" style="
                    flex: 1;
                    background-color: #4285F4;
                    color: #FFFFFF;
                    border: none;
                    padding: 12px;
                    border-radius: 6px;
                    text-align: center;
                    text-decoration: none;
                    font-weight: 500;
                    cursor: pointer;
                " onclick="closeModal()">Post to Google</a>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes modalFadeIn {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    
    <script>
    function showModal() {{
        document.getElementById('google-review-modal').style.display = 'flex';
        // Prevent scrolling of the page behind the modal
        document.body.style.overflow = 'hidden';
    }}
    
    function closeModal() {{
        document.getElementById('google-review-modal').style.display = 'none';
        // Re-enable scrolling
        document.body.style.overflow = 'auto';
    }}
    
    // Show the modal with a small delay
    setTimeout(showModal, 500);
    </script>
    """
    
    return modal_html

def trigger_google_review_popup(review_data):
    """
    Triggers the Google Review popup modal after saving a review.
    
    Args:
        review_data: The review data dictionary
    """
    # Get Google Place ID for the business
    PLACE_ID = os.environ.get("GOOGLE_PLACE_ID", "YOUR_PLACE_ID_HERE")
    
    # Create the modal popup HTML
    modal_html = create_google_review_modal(review_data, PLACE_ID)
    
    # Inject the modal HTML
    st.markdown(modal_html, unsafe_allow_html=True)
    
    # Return True to indicate the popup was triggered
    return True

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
        if "disable_state_validation" not in st.session_state:
            st.session_state.disable_state_validation = True  # Set to False in production

        # Add this at the beginning of your main() function
        def check_environment_variables():
            """Check if all required environment variables are set"""
            required_vars = {
                "GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
                "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
                "GOOGLE_REDIRECT_URI": GOOGLE_REDIRECT_URI,
                "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
                "LELAPA_API_TOKEN": os.environ.get("LELAPA_API_TOKEN"),
            }
            
            missing_vars = [var for var, value in required_vars.items() if not value]
            
            if missing_vars:
                st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
                st.info("Please add these variables to your .env file or environment.")
                
                # Show current configuration
                st.expander("Current Environment Configuration", expanded=True)
                for var, value in required_vars.items():
                    masked_value = "✓ Set" if value else "❌ Missing"
                    if value and var.endswith(("SECRET", "KEY", "TOKEN")):
                        masked_value = f"✓ Set (length: {len(value)})"
                    st.write(f"{var}: {masked_value}")
                
                return False
            return True

        # Call this at the beginning of main()
        if not check_environment_variables():
            st.stop()

        # Debug environment variables
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            st.error("Google OAuth credentials are not properly configured. Please check your .env file.")
            st.write("Debug - GOOGLE_CLIENT_ID set:", bool(GOOGLE_CLIENT_ID))
            st.write("Debug - GOOGLE_CLIENT_SECRET set:", bool(GOOGLE_CLIENT_SECRET))
            st.write("Debug - GOOGLE_REDIRECT_URI:", GOOGLE_REDIRECT_URI)
        load_css()
        
        # Handle OAuth callback if 'code' is in the query params
        if "code" in st.query_params:
            code = st.query_params["code"]
            st.write("Debug - OAuth code received")
            
            # Verify state parameter if available
            expected_state = st.session_state.get("oauth_state")
            received_state = st.query_params.get("state")
            
            if expected_state and received_state and expected_state != received_state:
                st.error("Invalid state parameter. Authentication failed.")
                # Clear params and return
                st.query_params.clear()
            else:
                # Process the callback
                if process_oauth_callback(code):
                    st.success("Authentication successful!")
                    # Clear URL parameters
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("Authentication failed. Please try again.")
                    # Clear parameters anyway
                    st.query_params.clear()
        
        ensure_directories()        
        # Sidebar with improved styling
        with st.sidebar:
            st.header("About")
            st.markdown(f"""
            <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                 box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;">
                <p style="margin-top: 0; color: #E0E0E0;">I'd love to hear about your experience with us! Your feedback helps me make things even better.</p>
                <p style="margin-bottom: 10px; color: #E0E0E0;">Here's how it works:</p>
                <ol style="padding-left: 20px; margin-bottom: 0; color: #E0E0E0;">
                    <li style="margin-bottom: 8px;">Record your thoughts (up to 25 seconds)</li>
                    <li style="margin-bottom: 8px;">I'll transcribe what you said</li>
                    <li style="margin-bottom: 8px;">Your feedback gets analyzed</li>
                    <li style="margin-bottom: 0;">You can share it on Google Reviews too!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # User account section with improved styling
            st.header("Your Account")
            
            # If user is logged in, show user info and logout button
            if check_login_status():
                # Display Google profile picture if available
                profile_pic = ""
                if st.session_state.current_user.get('picture'):
                    profile_pic = f"""<img src="{st.session_state.current_user.get('picture')}" 
                                       style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px;">"""
                else:
                    profile_pic = f"""<div style="width: 50px; height: 50px; border-radius: 50%; background-color: {BRAND_COLOR}; 
                                       color: black; display: flex; align-items: center; justify-content: center; 
                                       font-size: 20px; margin-right: 15px;">
                                       {st.session_state.current_user.get('name', 'U')[0].upper()}
                                   </div>"""
                
                google_badge = ""
                if 'google_id' in st.session_state.current_user:
                    google_badge = """<span style="background-color: #4285F4; color: white; padding: 3px 8px; 
                                     border-radius: 10px; font-size: 12px; margin-left: 10px;">Google</span>"""
                
                st.markdown(f"""
                <div style="background-color: #333333; padding: 20px; border-radius: 10px; 
                     box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        {profile_pic}
                        <div>
                            <p style="margin: 0; font-weight: 600; font-size: 18px; color: #FFFFFF;">
                                {st.session_state.current_user.get('name', 'User')}{google_badge}
                            </p>
                            <p style="margin: 0; color: #BBBBBB; font-size: 14px;">
                                {st.session_state.current_user.get('email', 'N/A')}
                            </p>
                        </div>
                    </div>
                    <p style="margin: 0; font-size: 14px; color: #BBBBBB;">
                        Last visit: {format_date(st.session_state.current_user.get('last_login', 'Unknown'))}
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
            # Only show logo without the title on login page
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image("Vintage Colorful Retro Vibes Typographic Product Brand Logo.png", width=80)
            
            # User is not logged in, show login form
            render_login_form()
            
            if st.session_state.register_success:
                st.success(st.session_state.register_success)
        else:
            # User is logged in - show full header with title
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image("Vintage Colorful Retro Vibes Typographic Product Brand Logo.png", width=80)
            with col2:
                st.title("My Restaurant Feedback Portal")
            
            # User is logged in, show main content with improved tab styling
            if st.session_state.is_owner:
                # Show all tabs for owner
                tab1, tab2, tab3 = st.tabs(["📝 Leave Feedback", "📋 All Feedback", "👤 My Feedback"])
            else:
                # Show only Leave Feedback and My Feedback tabs for regular users
                tab1, tab3 = st.tabs(["📝 Leave Feedback", "👤 My Feedback"])

            # Feedback tab
            with tab1:
                st.markdown(f"""
                <h2 style="color: #FFFFFF; margin-bottom: 20px;">
                    <span style="margin-right: 10px;">📝</span> Tell Me About Your Experience
                </h2>
                <p style="font-size: 16px; margin-bottom: 20px; color: #E0E0E0;">
                    I value your honest feedback! Let me know what you enjoyed and what I could improve.
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
                            Just speak into your mic - it's the quickest way to share your thoughts!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show recording UI if not showing analysis
                    if not st.session_state.show_analysis:
                        audio_file = record_audio()
                    
                    with st.expander("Having Audio Problems?"):
                        st.markdown("""
                        ### Microphone Access
                        - **Mobile**: Check your browser has mic permissions
                        - **Desktop**: Check your sound settings and browser permissions
                        - **iOS devices**: Safari works best for mic access
                        - **Android**: Make sure mic access is enabled
                        
                        ### Audio Quality Tips
                        - Find a quieter spot if possible
                        - Speak a bit louder than normal
                        - Don't cover your mic
                        - On mobile, keep the bottom of your device clear
                        
                        ### Troubleshooting
                        - Try refreshing the page
                        - Try Chrome or Safari if other browsers don't work
                        - You can always use the text input option instead
                        """)

                # Display analysis and save options
                if st.session_state.show_analysis and st.session_state.current_analysis:
                    with col1:
                        post_to_google = display_analysis(st.session_state.current_analysis)
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
                                Prefer typing? Share your experience in the box below.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        text_feedback = st.text_area("Your thoughts", height=200, 
                                                   placeholder="Tell me what you liked or didn't like... What made your experience special? What could I improve?")
                        
                        if st.button("📝 Submit Feedback", type="primary", use_container_width=True):
                            if text_feedback:
                                with st.spinner("Analyzing your feedback..."):
                                    review_analysis, validation_error = process_and_validate_review(text_feedback)
                                
                                if validation_error:
                                    st.error(validation_error)
                                    st.info("Could you share more details about your restaurant experience?")
                                elif review_analysis:
                                    st.success("Analysis complete!")
                                    st.session_state.current_analysis = review_analysis
                                    st.session_state.show_analysis = True
                                    st.rerun()
                            else:
                                st.warning("Please share your thoughts before submitting.")

            # View all feedback tab with improved styling
            if st.session_state.is_owner:
                with tab2:
                    st.markdown(f"""
                    <h2 style="color: #FFFFFF; margin-bottom: 20px;">
                        <span style="margin-right: 10px;">📋</span> All Customer Feedback
                    </h2>
                    <p style="font-size: 16px; margin-bottom: 20px; color: #E0E0E0;">
                        See what your customers are saying and find ways to improve.
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
                    <span style="margin-right: 10px;">👤</span> Your Feedback History
                </h2>
                <p style="font-size: 16px; margin-bottom: 20px; color: #E0E0E0;">
                    Here's all the feedback you've shared with me.
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
                    st.info("You haven't shared any feedback yet. I'd love to hear your thoughts!")
                else:
                    # Display count with message about limit
                    st.markdown(f"""
                    <div style="background-color: #333333; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <p style="margin: 0; font-size: 16px; color: #E0E0E0;">
                    """, unsafe_allow_html=True)
                    
                    if total_user_reviews_count > 10:
                        st.markdown(f"Showing your 10 most recent reviews (out of {total_user_reviews_count} total)")
                    else:
                        st.markdown(f"You've shared {total_user_reviews_count} reviews with me. Thank you!")
                    
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

        # Footer with improved styling
        st.markdown(f"""
        <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #444444;">
            <p style="color: #BBBBBB; font-size: 14px;">
                Thank you for sharing your thoughts with me! • © {datetime.now().year} My Restaurant
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Oops! Something went wrong: {str(e)}")
        st.info("Please refresh the page and try again. If the problem persists, please let me know.")

if __name__ == "__main__":
    main()
