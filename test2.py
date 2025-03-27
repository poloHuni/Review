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
from datetime import datetime
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Database functions
def load_reviews_from_db():
    try:
        if os.path.exists("restaurant_reviews.xlsx"):
            return pd.read_excel("restaurant_reviews.xlsx").to_dict(orient='records')
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

# Initialize session state variables
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

# LLM functions
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
    recorder_container = st.empty()
    process_container = st.empty()
    
    # Handle "Record Again" flow
    if st.session_state.record_again:
        st.session_state.record_again = False
        st.rerun()
    
    # Show instruction
    instruction_container.markdown("""
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 10px;">
        <p>🎙️ Click the microphone to start/stop recording your feedback (max 25 sec)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show recorder
    with recorder_container:
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e53935",
            neutral_color="#5a7d7c",
            icon_name="microphone",
            pause_threshold=25.0
        )
    
    # Process recorded audio
    if audio_bytes:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"review_{st.session_state.customer_id}_{timestamp}.wav"
        
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        
        st.session_state.audio_file = filename
        instruction_container.success("Recording completed!")
        
        # Process button
        if process_container.button("✅ Process Recording", key="process_audio_btn"):
            with st.spinner("Transcribing your feedback..."):
                transcribed_text = transcribe_audio(filename)
                
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
        if st.button("🔄 Record Again", key="record_again_btn"):
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
                    
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
    You are analyzing a customer's restaurant feedback. Please summarize this feedback, 
    extracting key points about the experience, food quality, service, atmosphere, and any 
    specific recommendations or complaints. Also rate the overall sentiment on a scale of 1-5.
    
    Customer feedback: {transcribed_text}
    
    Provide your analysis in the following JSON format. Make sure to escape any quotes within the text fields:
    {{
        "summary": "Brief summary of the overall experience",
        "food_quality": "Assessment of food quality",
        "service": "Assessment of service quality",
        "atmosphere": "Assessment of restaurant atmosphere",
        "specific_points": ["Point 1", "Point 2", "..."],
        "sentiment_score": X,
        "improvement_suggestions": ["Suggestion 1", "Suggestion 2", "..."]
    }}
    
    Important: Ensure your response is ONLY the valid JSON object, with no additional text before or after.
    """
    
    response = st.session_state.conversation.predict(input=prompt)
    
    try:
        # Extract JSON
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, response)
        
        if match:
            parsed_response = json.loads(match.group(1))
        else:
            parsed_response = json.loads(response)
            
        # Add raw transcription
        parsed_response["raw_transcription"] = transcribed_text
        return parsed_response
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse LLM response as JSON: {str(e)}")
        return {
            "summary": "Error processing review",
            "raw_transcription": transcribed_text,
            "error": "Failed to format response properly"
        }

# UI functions
def display_analysis(review_analysis):
    with st.container():
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h3 style="color: #5a7d7c;">Feedback Analysis</h3>
        """, unsafe_allow_html=True)
        
        st.write(f"**Summary**: {review_analysis.get('summary', 'N/A')}")
        st.write(f"**Food Quality**: {review_analysis.get('food_quality', 'N/A')}")
        st.write(f"**Service**: {review_analysis.get('service', 'N/A')}")
        st.write(f"**Atmosphere**: {review_analysis.get('atmosphere', 'N/A')}")
        
        # Sentiment indicator
        sentiment = review_analysis.get('sentiment_score', 'N/A')
        if isinstance(sentiment, (int, float)):
            st.write("**Overall Sentiment**:")
            sentiment_html = "".join(["⭐" for _ in range(int(sentiment))])
            st.markdown(f"<div style='font-size: 24px;'>{sentiment_html} ({sentiment}/5)</div>", unsafe_allow_html=True)
        else:
            st.write(f"**Overall Sentiment**: {sentiment}/5")
        
        # Display key points
        st.markdown("<h4 style='color: #5a7d7c; margin-top: 15px;'>Key Points</h4>", unsafe_allow_html=True)
        if 'specific_points' in review_analysis:
            for point in review_analysis['specific_points']:
                st.markdown(f"<div style='margin-left: 15px;'>• {point}</div>", unsafe_allow_html=True)
        
        # Display suggestions
        if 'improvement_suggestions' in review_analysis:
            st.markdown("<h4 style='color: #5a7d7c; margin-top: 15px;'>Suggestions for Improvement</h4>", unsafe_allow_html=True)
            for suggestion in review_analysis['improvement_suggestions']:
                st.markdown(f"<div style='margin-left: 15px;'>• {suggestion}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def collect_customer_info():
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #5a7d7c;">Personal Information </h3>
        <p>Providing your contact information helps us follow up on your feedback.</p>
    </div>
    """, unsafe_allow_html=True)
    
    customer_name = st.text_input("Name", value=st.session_state.customer_info["name"])
    customer_email = st.text_input("Email", value=st.session_state.customer_info["email"])
    customer_phone = st.text_input("Phone", value=st.session_state.customer_info["phone"])
    
    # Update customer info
    st.session_state.customer_info = {
        "name": customer_name, "email": customer_email, "phone": customer_phone
    }

# CSS for styling
def load_css():
    st.markdown("""
    <style>
    /* Custom styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #5a7d7c;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
    background-color: #5a7d7c;
    border-radius: 8px;
    color: black;
    padding: 10px 16px;
    font-size: 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #5a7d7c;
        color: white;
    }
    
    /* Card-like container style */
    .card-container {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Custom button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        border-color: #f7f7f7;
        color: #5a7d7c;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 16px;
        background-color: #f7f7f7;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Text area styling */
    .stTextArea>div>div {
        border-radius: 8px;
    }
    
    /* Customer info form styling */
    .customer-info-form {
        padding: 15px;
        border-radius: 10px;
        background-color: #f7f7f7;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

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

# Main application
def main():
    # Initialize state and styles
    init_session_state()
    load_css()
    
    st.title("Restaurant Feedback System")
    
    # Sidebar
    with st.sidebar:
        st.header("About this system")
        st.markdown("""
        <div class="card-container">
            <p>Share your dining experience with us! Your feedback helps us improve.</p>
            <p>This system will:</p>
            <ol>
                <li>Record your verbal feedback (25 seconds)</li>
                <li>Transcribe what you said</li>
                <li>Analyze your feedback</li>
                <li>Save your insights for the restaurant</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show recent reviews
        if st.session_state.reviews:
            st.header("Recent Feedback")
            recent_reviews = sorted(
                st.session_state.reviews, 
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )[:5]
            
            for i, review in enumerate(recent_reviews):
                display_date = format_date(review.get('timestamp', 'Unknown date'), "%b %d, %Y")
                
                with st.expander(f"Review {i+1} - {display_date}"):
                    st.write(f"**Summary**: {review.get('summary', 'N/A')}")
                    
                    # Display stars for sentiment
                    sentiment = review.get('sentiment_score', 'N/A')
                    if isinstance(sentiment, (int, float)):
                        sentiment_stars = "".join(["⭐" for _ in range(int(sentiment))])
                        st.markdown(f"**Sentiment**: {sentiment_stars} ({sentiment}/5)")
                    else:
                        st.write(f"**Sentiment**: {sentiment}/5")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📝 Leave Feedback", "📋 View All Feedback"])
    
    # Feedback tab
    with tab1:
        st.markdown("""
        <h2 style="color: #5a7d7c;">Share Your Experience</h2>
        <p>Please tell us about your dining experience at our restaurant.</p>
        """, unsafe_allow_html=True)
        
        # Collect customer information
        collect_customer_info()
            
        col1, col2 = st.columns(2)
        
        # Left column - Audio recording
        with col1:
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            # Show recording UI if not showing analysis
            if not st.session_state.show_analysis:
                audio_file = record_audio()
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Display analysis and save options
        if st.session_state.show_analysis and st.session_state.current_analysis:
            with col1:
                display_analysis(st.session_state.current_analysis)
                
                # Save/Cancel buttons
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾 Save Feedback"):
                        if save_review(st.session_state.current_analysis):
                            st.success("Thank you! Your feedback has been saved.")
                            # Reset states
                            st.session_state.audio_file = None
                            st.session_state.show_analysis = False
                            st.session_state.current_analysis = None
                            st.session_state.is_recording = False
                            st.rerun()
                
                with col_cancel:
                    if st.button("↩️ Start Over"):
                        # Reset states
                        st.session_state.audio_file = None
                        st.session_state.show_analysis = False
                        st.session_state.current_analysis = None
                        st.session_state.is_recording = False
                        st.session_state.record_again = True
                        st.rerun()

        # Text input - Right column
        with col2:
            if not st.session_state.show_analysis:
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.write("**Or type your feedback:**")
                text_feedback = st.text_area("Your feedback", height=150)
                
                if st.button("📝 Submit Written Feedback"):
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
                st.markdown('</div>', unsafe_allow_html=True)

    # View all feedback tab
    with tab2:
        st.markdown('<h2 style="color: #5a7d7c;">All Feedback</h2>', unsafe_allow_html=True)
        
        # Load latest reviews
        all_reviews = load_reviews_from_db()
        
        if not all_reviews:
            st.info("No feedback has been submitted yet.")
        else:
            # Sort reviews by timestamp
            sorted_reviews = sorted(all_reviews, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Filter controls
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            search_input = st.text_input("Search reviews by keyword:")
            
            # Sentiment filter
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
            st.markdown('</div>', unsafe_allow_html=True)
            
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
            
            # Display count
            st.write(f"Showing {len(filtered_reviews)} out of {len(all_reviews)} total reviews")
            
            # Display filtered reviews
            for review in filtered_reviews:
                display_date = format_date(review.get('timestamp', 'Unknown date'))

                with st.expander(f"Review from {display_date}"):
                    st.markdown('<div class="card-container">', unsafe_allow_html=True)
                    
                    # Customer info
                    if review.get('customer_name'):
                        st.write(f"**Customer**: {review['customer_name']}")
                    if review.get('customer_email'):
                        st.write(f"**Email**: {review['customer_email']}")
                    if review.get('customer_phone'):
                        st.write(f"**Phone**: {review['customer_phone']}")
                    
                    # Review details
                    st.write(f"**Summary**: {review.get('summary', 'N/A')}")
                    st.write(f"**Food Quality**: {review.get('food_quality', 'N/A')}")
                    st.write(f"**Service**: {review.get('service', 'N/A')}")
                    st.write(f"**Atmosphere**: {review.get('atmosphere', 'N/A')}")
                    
                    # Sentiment display
                    sentiment = review.get('sentiment_score', 'N/A')
                    if isinstance(sentiment, (int, float)):
                        sentiment_stars = "".join(["⭐" for _ in range(int(sentiment))])
                        st.markdown(f"**Sentiment Score**: {sentiment_stars} ({sentiment}/5)")
                    else:
                        st.write(f"**Sentiment Score**: {sentiment}/5")
                    
                    # Display key points
                    if 'specific_points' in review:
                        st.write("**Key Points:**")
                        display_list_items(review.get('specific_points', []))

                    # Display suggestions
                    if 'improvement_suggestions' in review:
                        st.write("**Suggestions for Improvement:**")
                        display_list_items(review.get('improvement_suggestions', []))
                        
                    # Show transcription
                    if 'raw_transcription' in review:
                        st.write("**Original Transcription:**")
                        st.text(review['raw_transcription'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()