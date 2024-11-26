# tool_control_system.py

# Import necessary libraries
import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, func, Index
from sqlalchemy.orm import sessionmaker, declarative_base
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pytesseract
import bcrypt
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime, timedelta
import logging
import calendar
import plotly.express as px
import itertools

# Initialize logging
logging.basicConfig(
    filename='tool_control.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set the Tesseract command path (ensure this path is correct for your system)
# For Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS/Linux (assuming tesseract is in PATH):
pytesseract.pytesseract.tesseract_cmd = r'tesseract'

# Database setup with SQLAlchemy
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    username = Column(String, primary_key=True)
    password = Column(String)
    role = Column(String)
    plant = Column(String)

class Plant(Base):
    __tablename__ = 'plants'
    plant_code = Column(String, primary_key=True)

class Tool(Base):
    __tablename__ = 'tools'
    tool_id = Column(String, primary_key=True)
    tool_name = Column(String)
    plant = Column(String, index=True)
    tool_type = Column(String, index=True)
    line_no = Column(String, index=True)
    zone = Column(String)
    serial_no = Column(Integer)
    operator_name = Column(String)
    epf_no = Column(String)
    status = Column(String, index=True)
    unique_symbol = Column(String)

class ToolIssuingLog(Base):
    __tablename__ = 'tool_issuing_log'
    id = Column(Integer, primary_key=True, autoincrement=True)
    tool_id = Column(String, index=True)
    operator_name = Column(String)
    epf_no = Column(String)
    issue_date = Column(DateTime, index=True)
    return_date = Column(DateTime, index=True)
    status = Column(String, index=True)
    issued_by = Column(String)
    plant = Column(String, index=True)
    line_no = Column(String)

class AuditLog(Base):
    __tablename__ = 'audit_log'
    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String)
    username = Column(String)
    timestamp = Column(DateTime, index=True)
    details = Column(Text)

# Create indexes for faster queries
Index('idx_tool_status', Tool.status)
Index('idx_tool_plant', Tool.plant)
Index('idx_log_issue_date', ToolIssuingLog.issue_date)
Index('idx_log_return_date', ToolIssuingLog.return_date)

engine = create_engine('sqlite:///tool_control.db', connect_args={'check_same_thread': False})
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

# Password hashing with bcrypt
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password)

# User authentication with input validation
def login_user(username, password):
    if not username or not password:
        return None
    user = session.query(User).filter_by(username=username).first()
    if user and check_password(user.password, password):
        return user
    return None

# Barcode scanning function with error handling
def scan_barcode(image):
    try:
        barcodes = decode(image)
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            return barcode_data
    except Exception as e:
        logging.error(f"Error scanning barcode: {e}")
    return None

# OCR scanning function with improved accuracy
def scan_symbol(image):
    try:
        # Preprocess image for better OCR accuracy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize image to enhance small symbols
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Apply Gaussian Blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # Dilate and erode to remove noise
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        # Use pytesseract to do OCR on the processed image
        custom_config = r'--oem 3 --psm 10'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        # Clean up the text
        text = text.strip()
        return text
    except Exception as e:
        logging.error(f"Error scanning symbol: {e}")
        return None

# Generate unique symbol with infinite combinations
def generate_unique_symbol():
    multilingual_chars = [
        # Greek letters
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο',
        'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
        # Cyrillic letters
        'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н',
        'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
        # Hebrew letters
        'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס',
        'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
        # Devanagari letters
        'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'च',
        'छ', 'ज', 'झ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ',
        'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह',
        # Latin letters
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        # Symbols
        '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+',
        '[', ']', '{', '}', '|', ';', ':', ',', '.', '<', '>', '/', '?'
    ]

    # Get all used symbols from the database
    used_symbols = set(tool.unique_symbol for tool in session.query(Tool.unique_symbol).all())

    # Initialize symbol length
    length = 1
    while True:
        # Generate all possible combinations of the given length
        combinations = itertools.product(multilingual_chars, repeat=length)
        for combination in combinations:
            symbol = ''.join(combination)
            if symbol not in used_symbols:
                return symbol
        # Increase the length if all combinations of the current length are used
        length += 1

# Generate symbol image with Noto Sans font
def generate_symbol_image(symbol):
    img_size = 200  # Increased size for better clarity
    img = Image.new('RGB', (img_size, img_size), color='white')
    draw = ImageDraw.Draw(img)
    # Use Noto Sans font
    font_path = 'NotoSans-Regular.ttf'  # Ensure this path is correct
    try:
        font_size = 150
        font = ImageFont.truetype(font_path, font_size)
    except OSError as e:
        st.error(f"Error loading font: {e}")
        return None
    # Calculate text size using textbbox
    bbox = draw.textbbox((0, 0), symbol, font=font)
    if bbox:
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_text = (img.width - text_width) / 2
        y_text = (img.height - text_height) / 2
        draw.text((x_text, y_text), symbol, font=font, fill='black')
        return img
    else:
        st.error("Failed to calculate text bounding box.")
        return None

# Generate barcode image with error handling
def generate_barcode(tool_id, operator_name, epf_no):
    from barcode import Code128
    from barcode.writer import ImageWriter

    try:
        barcode_image = Code128(tool_id, writer=ImageWriter())
        buffer = io.BytesIO()
        barcode_image.write(buffer)
        buffer.seek(0)
        img = Image.open(buffer)

        # Add text below barcode
        canvas_width, canvas_height = img.size
        canvas_height += 50  # Additional space for text
        new_img = Image.new('RGB', (canvas_width, canvas_height), 'white')
        new_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        text = f"Tool ID: {tool_id} | Operator: {operator_name} | EPF: {epf_no}"
        font_path = 'NotoSans-Regular.ttf'  # Use Noto Sans font
        try:
            font = ImageFont.truetype(font_path, 14)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_text = (canvas_width - text_width) / 2
        y_text = img.size[1] + 10
        draw.text((x_text, y_text), text, fill='black', font=font)

        return new_img
    except Exception as e:
        logging.error(f"Error generating barcode: {e}")
        return None

# Log user actions
def log_action(action, username, details):
    timestamp = datetime.now()
    audit_entry = AuditLog(action=action, username=username, timestamp=timestamp, details=details)
    session.add(audit_entry)
    session.commit()
    logging.info(f"{username} performed action: {action} - {details}")

# Process scanned image for barcode or symbol
def process_scanned_image(image, plant):
    scanned_tool_id = scan_barcode(image)
    if scanned_tool_id:
        st.success(f"Scanned Tool ID: {scanned_tool_id}")
        return scanned_tool_id
    else:
        scanned_symbol = scan_symbol(image)
        if scanned_symbol:
            # Map symbol to tool ID
            tool = session.query(Tool).filter_by(unique_symbol=scanned_symbol, plant=plant).first()
            if tool:
                st.success(f"Recognized Symbol: {scanned_symbol}, Tool ID: {tool.tool_id}")
                return tool.tool_id
            else:
                st.error("Symbol not recognized")
        else:
            st.error("No barcode or symbol detected")
    return None

# Main application
def main():
    st.set_page_config(page_title="Jay Jay Tool Control System (Beta))", layout="wide")

    # Add logo to sidebar
    logo_path = 'Black and Purple Gradient Modern Futuristic Rocket Icon Tech Logo (1).png'  
    if os.path.exists(logo_path):
        logo_image = Image.open(logo_path)
        st.sidebar.image(logo_image, use_column_width=True)

    st.title("Jay Jay Tool Control System (Trial)")

    # Session state for user authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.session_state.role = ''
        st.session_state.plant = ''
        st.session_state.session_expiry = datetime.now()

    if st.session_state.logged_in and datetime.now() < st.session_state.session_expiry:
        st.sidebar.write(f"Logged in as {st.session_state.username} ({st.session_state.role})")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.session_state.role = ''
            st.session_state.plant = ''
            st.experimental_rerun()
        else:
            # Extend session expiry
            st.session_state.session_expiry = datetime.now() + timedelta(minutes=30)
            if st.session_state.role == 'PlantRecorder':
                plant_recorder_ui()
            elif st.session_state.role == 'Stores':
                stores_ui()
            elif st.session_state.role == 'PlantAdmin':
                plant_admin_ui()
            elif st.session_state.role == 'MasterAdmin':
                master_admin_ui()
            else:
                st.error("Invalid role assigned.")
    else:
        st.session_state.logged_in = False
        menu = ["Login", "Master Admin SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            st.subheader("Login to your account")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Login"):
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = user.username
                    st.session_state.role = user.role
                    st.session_state.plant = user.plant
                    st.session_state.session_expiry = datetime.now() + timedelta(minutes=30)
                    log_action('Login', user.username, f"User {user.username} logged in.")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
                    log_action('Failed Login', username, "Invalid credentials.")

        elif choice == "Master Admin SignUp":
            st.subheader("Create a Master Admin account")
            new_user = st.text_input("Username", key="ma_signup_username")
            new_password = st.text_input("Password", type='password', key="ma_signup_password")
            confirm_password = st.text_input("Confirm Password", type='password', key="ma_signup_confirm_password")
            if st.button("SignUp", key="ma_signup_button"):
                if new_user and new_password and confirm_password:
                    if new_password == confirm_password:
                        existing_user = session.query(User).filter_by(username=new_user).first()
                        if existing_user:
                            st.error("Username already exists.")
                        else:
                            # Enforce strong password policies
                            if len(new_password) < 8:
                                st.error("Password must be at least 8 characters long.")
                            elif not any(char.isdigit() for char in new_password):
                                st.error("Password must contain at least one number.")
                            elif not any(char.isupper() for char in new_password):
                                st.error("Password must contain at least one uppercase letter.")
                            else:
                                hashed_pwd = hash_password(new_password)
                                new_user_obj = User(username=new_user, password=hashed_pwd, role='MasterAdmin', plant='All')
                                session.add(new_user_obj)
                                session.commit()
                                st.success("Master Admin account created successfully")
                                st.info("Go to Login Menu to login")
                                log_action('SignUp', new_user, "Created Master Admin account.")
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.error("Please fill in all fields.")

# Plant Recorder UI
def plant_recorder_ui():
    username = st.session_state.username
    plant = st.session_state.plant
    st.header(f"Plant Recorder Dashboard - Plant {plant}")
    tabs = st.tabs(["Receive Tools from Stores", "Issue Tool to Operator", "Return Tool from Operator", "Search Tool", "Dashboard"])

    # Receive Tools from Stores
    with tabs[0]:
        st.subheader("Receive Tools from Stores")
        tool_id = st.text_input("Tool ID (Scan Barcode or Symbol)", key="pr_receive_tool_id")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.camera_input("Scan with Camera", key="pr_receive_camera_input")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        with col2:
            image_file = st.file_uploader("Upload Image for Scanning", type=["png", "jpg", "jpeg"], key="pr_receive_file_upload")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        if st.button("Receive Tool from Stores", key="pr_receive_button"):
            if tool_id:
                tool = session.query(Tool).filter_by(tool_id=tool_id, plant=plant, status='Available').first()
                if tool:
                    tool.status = 'Issued to Plant Recorder'
                    session.commit()
                    st.success(f"Tool {tool_id} received from Stores")
                    log_action('Receive Tool from Stores', username, f"Received tool {tool_id} from Stores")
                else:
                    st.error("Tool not available or already issued.")
            else:
                st.error("Please enter the Tool ID.")

    # Issue Tool to Operator
    with tabs[1]:
        st.subheader("Issue Tool to Operator")
        operator_name = st.text_input("Operator Name", key="pr_issue_operator_name")
        epf_no = st.text_input("EPF Number", key="pr_issue_epf_no")
        tool_id = st.text_input("Tool ID (Scan Barcode or Symbol)", key="pr_issue_tool_id")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.camera_input("Scan with Camera", key="pr_issue_camera_input")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        with col2:
            image_file = st.file_uploader("Upload Image for Scanning", type=["png", "jpg", "jpeg"], key="pr_issue_file_upload")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        if st.button("Issue Tool to Operator", key="pr_issue_button"):
            if tool_id and operator_name and epf_no:
                tool = session.query(Tool).filter_by(tool_id=tool_id, plant=plant, status='Issued to Plant Recorder').first()
                if tool:
                    issue_date = datetime.now()
                    new_issue = ToolIssuingLog(
                        tool_id=tool_id,
                        operator_name=operator_name,
                        epf_no=epf_no,
                        issue_date=issue_date,
                        status='Issued to Operator',
                        issued_by=username,
                        plant=plant,
                        line_no=tool.line_no
                    )
                    session.add(new_issue)
                    tool.status = 'Issued to Operator'
                    session.commit()
                    st.success(f"Tool {tool_id} issued to {operator_name}")
                    log_action('Issue Tool to Operator', username, f"Issued tool {tool_id} to {operator_name}")
                else:
                    st.error("Tool not available for issuing.")
            else:
                st.error("Please fill in all fields.")

    # Return Tool from Operator
    with tabs[2]:
        st.subheader("Return Tool from Operator")
        tool_id = st.text_input("Tool ID (Scan Barcode or Symbol)", key="pr_return_tool_id")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.camera_input("Scan with Camera", key="pr_return_camera_input")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        with col2:
            image_file = st.file_uploader("Upload Image for Scanning", type=["png", "jpg", "jpeg"], key="pr_return_file_upload")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        if st.button("Return Tool from Operator", key="pr_return_button"):
            if tool_id:
                return_date = datetime.now()
                issue_record = session.query(ToolIssuingLog).filter_by(
                    tool_id=tool_id,
                    status='Issued to Operator',
                    plant=plant
                ).first()
                if issue_record:
                    issue_record.return_date = return_date
                    issue_record.status = 'Returned by Operator'
                    tool = session.query(Tool).filter_by(tool_id=tool_id, plant=plant).first()
                    tool.status = 'Issued to Plant Recorder'
                    session.commit()
                    st.success(f"Tool {tool_id} returned from operator")
                    log_action('Return Tool from Operator', username, f"Tool {tool_id} returned from operator")
                else:
                    st.error("No issued record found for this tool.")
            else:
                st.error("Please enter the Tool ID.")

    # Search Tool Information
    with tabs[3]:
        st.subheader("Search Tool Information")
        tool_id = st.text_input("Tool ID (Scan Barcode or Symbol)", key="pr_search_tool_id")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.camera_input("Scan with Camera", key="pr_search_camera_input")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        with col2:
            image_file = st.file_uploader("Upload Image for Scanning", type=["png", "jpg", "jpeg"], key="pr_search_file_upload")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        if st.button("Search", key="pr_search_button"):
            if tool_id:
                records = session.query(ToolIssuingLog).filter_by(tool_id=tool_id, plant=plant).all()
                if records:
                    df = pd.DataFrame([r.__dict__ for r in records])
                    df.drop('_sa_instance_state', axis=1, inplace=True)
                    st.dataframe(df)
                    log_action('Search Tool', username, f"Searched for tool {tool_id}")
                else:
                    st.info("No records found for this tool.")
            else:
                st.error("Please enter the Tool ID.")

    # Dashboard
    with tabs[4]:
        st.subheader("Dashboard")
        # Number Cards
        total_tools = session.query(func.count(Tool.tool_id)).filter_by(plant=plant).scalar()
        tools_with_recorder = session.query(func.count(Tool.tool_id)).filter_by(plant=plant, status='Issued to Plant Recorder').scalar()
        tools_with_operator = session.query(func.count(Tool.tool_id)).filter_by(plant=plant, status='Issued to Operator').scalar()
        available_tools = session.query(func.count(Tool.tool_id)).filter_by(plant=plant, status='Available').scalar()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tools", total_tools)
        with col2:
            st.metric("With Plant Recorder", tools_with_recorder)
        with col3:
            st.metric("With Operator", tools_with_operator)
        with col4:
            st.metric("Available in Stores", available_tools)

        # Visualizations
        st.subheader("Line-wise Tools Not Returned")
        data = session.query(Tool.line_no, func.count(Tool.tool_id)).filter(
            Tool.plant == plant,
            Tool.status == 'Issued to Operator'
        ).group_by(Tool.line_no).all()
        if data:
            df = pd.DataFrame(data, columns=['Line Number', 'Not Returned'])
            fig = px.bar(df, x='Line Number', y='Not Returned', title='Line-wise Tools Not Returned')
            st.plotly_chart(fig)
        else:
            st.info("No data available for visualization.")

        # Additional Charts for Detailed Analysis
        st.subheader("Tool Status Distribution")
        status_counts = session.query(Tool.status, func.count(Tool.tool_id)).filter_by(plant=plant).group_by(Tool.status).all()
        if status_counts:
            df_status = pd.DataFrame(status_counts, columns=['Status', 'Count'])
            fig = px.pie(df_status, names='Status', values='Count', title='Tool Status Distribution')
            st.plotly_chart(fig)

        st.subheader("Tools by Type")
        type_counts = session.query(Tool.tool_type, func.count(Tool.tool_id)).filter_by(plant=plant).group_by(Tool.tool_type).all()
        if type_counts:
            df_type = pd.DataFrame(type_counts, columns=['Tool Type', 'Count'])
            fig = px.bar(df_type, x='Tool Type', y='Count', title='Tools by Type')
            st.plotly_chart(fig)

# Stores UI
def stores_ui():
    username = st.session_state.username
    plant = st.session_state.plant
    st.header(f"Stores Dashboard - Plant {plant}")
    tabs = st.tabs(["Add New Tool", "Issue Tool to Plant Recorder", "Total Inventory", "Dashboard", "Search Tool", "Plant Tool Reconciliation Report"])

    # Add New Tool to Inventory
    with tabs[0]:
        st.subheader("Add New Tool to Inventory")
        tool_type_list = ["Trimmers", "Scissors", "Tag guns", "Button pins", "Twissors", "Shuttle bobbins", "O/L Pins",
                          "Metal gloves", "Layering clips", "Layering pins", "Cello tape dispensers", "Punchers", "Pointers"]
        tool_type = st.selectbox("Tool Type", tool_type_list, key="stores_tool_type")
        line_no = st.text_input("Line Number", key="stores_line_no")
        zone = st.text_input("Zone", key="stores_zone")

        if st.button("Generate Tool ID", key="stores_generate_tool_id"):
            if line_no and zone:
                # Auto-generate serial number
                max_serial_no = session.query(func.max(Tool.serial_no)).filter_by(
                    plant=plant,
                    tool_type=tool_type,
                    line_no=line_no,
                    zone=zone
                ).scalar()
                if max_serial_no is None:
                    max_serial_no = 0
                serial_no = max_serial_no + 1
                st.session_state.serial_no = serial_no
                tool_type_initial = tool_type[:3].upper()
                tool_id = f"{plant}/{tool_type_initial}/{line_no}/{zone}/{serial_no}"
                st.write(f"Generated Tool ID: {tool_id}")
                st.session_state.tool_id = tool_id
            else:
                st.error("Please fill in all fields.")

        if 'tool_id' in st.session_state:
            tool_id = st.session_state.tool_id
            serial_no = st.session_state.serial_no
            operator_name = st.text_input("Operator Name for Barcode", key="stores_operator_name")
            epf_no = st.text_input("EPF Number for Barcode", key="stores_epf_no")
            if st.button("Add Tool", key="stores_add_tool"):
                if operator_name and epf_no:
                    existing_tool = session.query(Tool).filter_by(tool_id=tool_id).first()
                    if existing_tool:
                        st.error("Tool ID already exists.")
                    else:
                        unique_symbol = generate_unique_symbol()
                        if not unique_symbol:
                            st.error("Failed to generate a unique symbol. Please contact the administrator.")
                            return
                        new_tool = Tool(
                            tool_id=tool_id,
                            tool_name=tool_type,
                            plant=plant,
                            tool_type=tool_type,
                            line_no=line_no,
                            zone=zone,
                            serial_no=serial_no,
                            operator_name=operator_name,
                            epf_no=epf_no,
                            status='Available',
                            unique_symbol=unique_symbol
                        )
                        session.add(new_tool)
                        session.commit()
                        st.success(f"Tool {tool_type} with ID {tool_id} added to inventory")
                        log_action('Add New Tool', username, f"Added tool {tool_id}")

                        # Generate barcode
                        barcode_image = generate_barcode(tool_id, operator_name, epf_no)
                        # Generate symbol image
                        symbol_image = generate_symbol_image(unique_symbol)
                        st.image(barcode_image, caption='Generated Barcode', use_column_width=True)
                        st.image(symbol_image, caption=f'Unique Symbol: {unique_symbol}', use_column_width=False)

                        # Option to download barcode
                        buf = io.BytesIO()
                        barcode_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        b64 = base64.b64encode(byte_im).decode()
                        href = f'<a href="data:file/png;base64,{b64}" download="barcode_{tool_id}.png">Download Barcode</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Option to download symbol image
                        buf = io.BytesIO()
                        symbol_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        b64 = base64.b64encode(byte_im).decode()
                        href = f'<a href="data:file/png;base64,{b64}" download="symbol_{tool_id}.png">Download Symbol</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("Please enter Operator Name and EPF Number for the barcode.")
        else:
            st.info("Click 'Generate Tool ID' to proceed.")

    # Issue Tool to Plant Recorder
    with tabs[1]:
        st.subheader("Issue Tool to Plant Recorder")
        tool_id = st.text_input("Tool ID (Scan Barcode or Symbol)", key="stores_issue_tool_id")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.camera_input("Scan with Camera", key="stores_issue_camera_input")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        with col2:
            image_file = st.file_uploader("Upload Image for Scanning", type=["png", "jpg", "jpeg"], key="stores_issue_file_upload")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        if st.button("Issue Tool to Plant Recorder", key="stores_issue_button"):
            if tool_id:
                tool = session.query(Tool).filter_by(tool_id=tool_id, plant=plant, status='Available').first()
                if tool:
                    tool.status = 'Issued to Plant Recorder'
                    session.commit()
                    st.success(f"Tool {tool_id} issued to Plant Recorder")
                    log_action('Issue Tool to Plant Recorder', username, f"Issued tool {tool_id} to Plant Recorder")
                else:
                    st.error("Tool not available or already issued.")
            else:
                st.error("Please enter the Tool ID.")

    # Total Inventory
    with tabs[2]:
        st.subheader("Total Inventory")
        tools = session.query(Tool).filter_by(plant=plant).all()
        if tools:
            df = pd.DataFrame([t.__dict__ for t in tools])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)
            log_action('View Inventory', username, f"Viewed inventory for plant {plant}")
        else:
            st.info("No tools found in the inventory.")

    # Dashboard
    with tabs[3]:
        st.subheader("Dashboard")
        # Number Cards
        total_tools = session.query(func.count(Tool.tool_id)).filter_by(plant=plant).scalar()
        tools_with_recorder = session.query(func.count(Tool.tool_id)).filter_by(plant=plant, status='Issued to Plant Recorder').scalar()
        available_tools = session.query(func.count(Tool.tool_id)).filter_by(plant=plant, status='Available').scalar()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tools", total_tools)
        with col2:
            st.metric("With Plant Recorder", tools_with_recorder)
        with col3:
            st.metric("Available in Stores", available_tools)

        # Visualizations
        st.subheader("Tool Distribution by Type")
        tool_counts = session.query(Tool.tool_type, func.count(Tool.tool_id)).filter_by(plant=plant).group_by(Tool.tool_type).all()
        if tool_counts:
            df = pd.DataFrame(tool_counts, columns=['Tool Type', 'Count'])
            fig = px.pie(df, names='Tool Type', values='Count', title='Tool Distribution by Type')
            st.plotly_chart(fig)
        else:
            st.info("No data to display.")

        # Additional Charts
        st.subheader("Tool Status Distribution")
        status_counts = session.query(Tool.status, func.count(Tool.tool_id)).filter_by(plant=plant).group_by(Tool.status).all()
        if status_counts:
            df_status = pd.DataFrame(status_counts, columns=['Status', 'Count'])
            fig = px.bar(df_status, x='Status', y='Count', title='Tool Status Distribution')
            st.plotly_chart(fig)

    # Search Tool Information
    with tabs[4]:
        st.subheader("Search Tool Information")
        tool_id = st.text_input("Tool ID (Scan Barcode or Symbol)", key="stores_search_tool_id")
        col1, col2 = st.columns(2)
        with col1:
            image_file = st.camera_input("Scan with Camera", key="stores_search_camera_input")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        with col2:
            image_file = st.file_uploader("Upload Image for Scanning", type=["png", "jpg", "jpeg"], key="stores_search_file_upload")
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
                tool_id = process_scanned_image(image, plant)
        if st.button("Search", key="stores_search_button"):
            if tool_id:
                records = session.query(Tool).filter_by(tool_id=tool_id, plant=plant).all()
                if records:
                    df = pd.DataFrame([r.__dict__ for r in records])
                    df.drop('_sa_instance_state', axis=1, inplace=True)
                    st.dataframe(df)
                    log_action('Search Tool', username, f"Searched for tool {tool_id}")
                else:
                    st.info("No records found for this tool.")
            else:
                st.error("Please enter the Tool ID.")

    # Plant Tool Reconciliation Report
    with tabs[5]:
        st.subheader("Plant Tool Reconciliation Report")
        # Option to select month
        month = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: calendar.month_name[x], key="stores_recon_month")
        year = st.number_input("Enter Year", min_value=2000, max_value=datetime.now().year, value=datetime.now().year, key="stores_recon_year")

        start_date = datetime(year, month, 1)
        end_date = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59)

        records = session.query(ToolIssuingLog).filter(
            ToolIssuingLog.plant == plant,
            ToolIssuingLog.issue_date >= start_date,
            ToolIssuingLog.issue_date <= end_date
        ).all()

        if records:
            df = pd.DataFrame([r.__dict__ for r in records])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)
            log_action('View Reconciliation Report', username, f"Viewed reconciliation report for plant {plant}")

            # Option to download report
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f'plant_{plant}_reconciliation_report_{year}_{month}.csv',
                mime='text/csv',
                key="stores_download_csv"
            )

            # Additional Charts
            st.subheader("Tools Issued and Returned")
            df['issue_date'] = pd.to_datetime(df['issue_date'])
            df['return_date'] = pd.to_datetime(df['return_date'])
            df['Duration'] = (df['return_date'] - df['issue_date']).dt.days
            duration_df = df[['issue_date', 'Duration']].dropna()
            if not duration_df.empty:
                fig = px.histogram(duration_df, x='Duration', nbins=10, title='Distribution of Tool Usage Duration (Days)')
                st.plotly_chart(fig)
        else:
            st.info("No data available for the selected month.")

# Plant Admin UI
def plant_admin_ui():
    username = st.session_state.username
    plant = st.session_state.plant
    st.header(f"Plant Admin Dashboard - Plant {plant}")
    tabs = st.tabs(["Stores Functions", "Plant Recorder Functions", "Reports", "Audit Log", "Data Analysis"])
    with tabs[0]:
        st.subheader("Stores Functions")
        stores_ui()
    with tabs[1]:
        st.subheader("Plant Recorder Functions")
        plant_recorder_ui()
    with tabs[2]:
        st.subheader("Reports")
        # Option to select month
        month = st.selectbox("Select Month for Report", list(range(1, 13)), format_func=lambda x: calendar.month_name[x], key="pa_report_month")
        year = st.number_input("Enter Year for Report", min_value=2000, max_value=datetime.now().year, value=datetime.now().year, key="pa_report_year")

        start_date = datetime(year, month, 1)
        end_date = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59)

        records = session.query(ToolIssuingLog).filter(
            ToolIssuingLog.plant == plant,
            ToolIssuingLog.issue_date >= start_date,
            ToolIssuingLog.issue_date <= end_date
        ).all()

        if records:
            df = pd.DataFrame([r.__dict__ for r in records])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)

            # Option to download report
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f'plant_{plant}_report_{year}_{month}.csv',
                mime='text/csv',
                key="pa_download_csv"
            )

            # Visualizations with filters
            st.subheader("Interactive Data Visualization")
            status_filter = st.multiselect("Select Status", options=df['status'].unique(), default=df['status'].unique(), key="pa_status_filter")
            line_filter = st.multiselect("Select Line Number", options=df['line_no'].unique(), default=df['line_no'].unique(), key="pa_line_filter")
            filtered_df = df[df['status'].isin(status_filter) & df['line_no'].isin(line_filter)]

            st.write("Filtered Data", filtered_df)

            st.subheader("Status Distribution")
            status_counts = filtered_df['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig = px.bar(status_counts, x='Status', y='Count', title='Status Distribution')
            st.plotly_chart(fig)

            st.subheader("Tools Issued Over Time")
            filtered_df['issue_date'] = pd.to_datetime(filtered_df['issue_date'])
            issue_counts = filtered_df.groupby(filtered_df['issue_date'].dt.date).size().reset_index(name='Counts')
            fig = px.line(issue_counts, x='issue_date', y='Counts', title='Tools Issued Over Time')
            st.plotly_chart(fig)

            st.subheader("Average Tool Usage Duration")
            filtered_df['return_date'] = pd.to_datetime(filtered_df['return_date'])
            filtered_df['Duration'] = (filtered_df['return_date'] - filtered_df['issue_date']).dt.days
            avg_duration = filtered_df['Duration'].mean()
            st.metric("Average Duration (Days)", f"{avg_duration:.2f}")
            log_action('View Reports', username, f"Viewed reports for plant {plant}")
        else:
            st.info("No data available for the selected month.")

    with tabs[3]:
        st.subheader("Audit Log")
        audit_entries = session.query(AuditLog).filter(AuditLog.username.in_(
            session.query(User.username).filter_by(plant=plant)
        )).all()
        if audit_entries:
            df = pd.DataFrame([a.__dict__ for a in audit_entries])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)
        else:
            st.info("No audit logs available.")

    with tabs[4]:
        st.subheader("Data Analysis Dashboard")
        # Comprehensive visualizations with filters
        records = session.query(ToolIssuingLog).filter_by(plant=plant).all()
        if records:
            df = pd.DataFrame([r.__dict__ for r in records])
            df.drop('_sa_instance_state', axis=1, inplace=True)

            st.subheader("Filter Options")
            status_filter = st.multiselect("Select Status", options=df['status'].unique(), default=df['status'].unique(), key="pa_data_status_filter")
            line_filter = st.multiselect("Select Line Number", options=df['line_no'].unique(), default=df['line_no'].unique(), key="pa_data_line_filter")
            date_range = st.date_input("Select Date Range", [df['issue_date'].min(), df['issue_date'].max()], key="pa_data_date_range")

            filtered_df = df[
                (df['status'].isin(status_filter)) &
                (df['line_no'].isin(line_filter)) &
                (pd.to_datetime(df['issue_date']).dt.date >= date_range[0]) &
                (pd.to_datetime(df['issue_date']).dt.date <= date_range[1])
            ]

            st.write("Filtered Data", filtered_df)

            st.subheader("Tool Issuance by Operator")
            top_operators = filtered_df['operator_name'].value_counts().reset_index()
            top_operators.columns = ['Operator Name', 'Count']
            fig = px.bar(top_operators.head(10), x='Operator Name', y='Count', title='Top Operators by Tool Issuance')
            st.plotly_chart(fig)

            st.subheader("Issue and Return Timeline")
            filtered_df['Duration'] = (pd.to_datetime(filtered_df['return_date']) - pd.to_datetime(filtered_df['issue_date'])).dt.days
            duration_df = filtered_df[['issue_date', 'Duration']].dropna()
            fig = px.line(duration_df, x='issue_date', y='Duration', title='Issue and Return Timeline')
            st.plotly_chart(fig)

            # Additional Charts for Detailed Data Analysis
            st.subheader("Tool Usage Duration Distribution")
            if not duration_df.empty:
                fig = px.histogram(duration_df, x='Duration', nbins=10, title='Distribution of Tool Usage Duration (Days)')
                st.plotly_chart(fig)
        else:
            st.info("No data available for analysis.")

# Master Admin UI
def master_admin_ui():
    username = st.session_state.username
    st.header("Master Admin Dashboard")
    tabs = st.tabs(["Manage Plants", "View All Tool Data", "Download Reports", "Manage Users", "Audit Log"])
    with tabs[0]:
        st.subheader("Add or Delete Plants")
        plants = session.query(Plant).all()
        plant_codes = [p.plant_code for p in plants]
        st.write("Current Plants:", plant_codes)

        col1, col2 = st.columns(2)
        with col1:
            new_plant = st.text_input("New Plant Code", key="ma_new_plant")
            if st.button("Add Plant", key="ma_add_plant"):
                if new_plant:
                    existing_plant = session.query(Plant).filter_by(plant_code=new_plant).first()
                    if existing_plant:
                        st.error("Plant code already exists.")
                    else:
                        new_plant_obj = Plant(plant_code=new_plant)
                        session.add(new_plant_obj)
                        session.commit()
                        st.success(f"Plant {new_plant} added")
                        log_action('Add Plant', username, f"Added plant {new_plant}")
                else:
                    st.error("Please enter a Plant Code.")
        with col2:
            if plant_codes:
                delete_plant = st.selectbox("Select Plant to Delete", plant_codes, key="ma_delete_plant_select")
                if st.button("Delete Plant", key="ma_delete_plant"):
                    session.query(Plant).filter_by(plant_code=delete_plant).delete()
                    session.commit()
                    st.success(f"Plant {delete_plant} deleted")
                    log_action('Delete Plant', username, f"Deleted plant {delete_plant}")
            else:
                st.info("No plants available to delete.")

    with tabs[1]:
        st.subheader("All Tool Data Across Plants")
        records = session.query(ToolIssuingLog).all()
        if records:
            df = pd.DataFrame([r.__dict__ for r in records])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)
            log_action('View All Tool Data', username, "Viewed all tool data.")
        else:
            st.info("No tool data available.")

    with tabs[2]:
        st.subheader("Download Reports")
        # Option to select month
        month = st.selectbox("Select Month for Report", list(range(1, 13)), format_func=lambda x: calendar.month_name[x], key="ma_report_month")
        year = st.number_input("Enter Year for Report", min_value=2000, max_value=datetime.now().year, value=datetime.now().year, key="ma_report_year")

        start_date = datetime(year, month, 1)
        end_date = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59)

        records = session.query(ToolIssuingLog).filter(
            ToolIssuingLog.issue_date >= start_date,
            ToolIssuingLog.issue_date <= end_date
        ).all()

        if records:
            df = pd.DataFrame([r.__dict__ for r in records])
            df.drop('_sa_instance_state', axis=1, inplace=True)

            # Option to download report with barcodes and symbols in separate cells
            if st.button("Download Report with Barcodes and Symbols", key="ma_download_report"):
                excel_file = io.BytesIO()
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Tool Data')
                    workbook = writer.book
                    worksheet = writer.sheets['Tool Data']

                    # Adjust column width
                    worksheet.set_column('A:Z', 20)

                    # Add barcodes and symbols in separate cells
                    for idx, row in df.iterrows():
                        tool_id = row['tool_id']
                        operator_name = row['operator_name']
                        epf_no = row['epf_no']
                        unique_symbol = session.query(Tool.unique_symbol).filter_by(tool_id=tool_id).scalar()
                        barcode_image = generate_barcode(tool_id, operator_name, epf_no)
                        symbol_image = generate_symbol_image(unique_symbol)

                        if barcode_image and symbol_image:
                            # Save barcode image to a BytesIO object
                            barcode_stream = io.BytesIO()
                            barcode_image.save(barcode_stream, format='PNG')
                            barcode_stream.seek(0)

                            # Save symbol image to a BytesIO object
                            symbol_stream = io.BytesIO()
                            symbol_image.save(symbol_stream, format='PNG')
                            symbol_stream.seek(0)

                            # Insert images into separate cells
                            worksheet.insert_image(f'I{idx+2}', '', {'image_data': barcode_stream})
                            worksheet.insert_image(f'J{idx+2}', '', {'image_data': symbol_stream})

                excel_file.seek(0)
                st.download_button(
                    label="Download data as Excel",
                    data=excel_file,
                    file_name=f'all_tool_data_with_barcodes_and_symbols_{year}_{month}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key="ma_download_excel"
                )
                log_action('Download Report', username, "Downloaded report with barcodes and symbols.")
        else:
            st.info("No data available to download.")

    with tabs[3]:
        st.subheader("Manage Users")
        users = session.query(User).all()
        if users:
            df = pd.DataFrame([u.__dict__ for u in users])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)
        else:
            st.info("No users found.")

        st.subheader("Add New User")
        new_username = st.text_input("Username", key="ma_new_username")
        new_password = st.text_input("Password", type='password', key="ma_new_password")
        confirm_password = st.text_input("Confirm Password", type='password', key="ma_new_confirm_password")
        new_role = st.selectbox("Role", ["Stores", "PlantRecorder", "PlantAdmin"], key="ma_new_role")
        plants = session.query(Plant).all()
        plant_codes = [p.plant_code for p in plants]
        if plant_codes:
            new_plant = st.selectbox("Plant", plant_codes, key="ma_new_user_plant")
            if st.button("Create User", key="ma_create_user"):
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        existing_user = session.query(User).filter_by(username=new_username).first()
                        if existing_user:
                            st.error("Username already exists.")
                        else:
                            # Enforce strong password policies
                            if len(new_password) < 8:
                                st.error("Password must be at least 8 characters long.")
                            elif not any(char.isdigit() for char in new_password):
                                st.error("Password must contain at least one number.")
                            elif not any(char.isupper() for char in new_password):
                                st.error("Password must contain at least one uppercase letter.")
                            else:
                                hashed_pwd = hash_password(new_password)
                                new_user_obj = User(
                                    username=new_username,
                                    password=hashed_pwd,
                                    role=new_role,
                                    plant=new_plant
                                )
                                session.add(new_user_obj)
                                session.commit()
                                st.success(f"User {new_username} created with role {new_role} for plant {new_plant}")
                                log_action('Add User', username, f"Added user {new_username}")
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.error("Please fill in all fields.")
        else:
            st.info("No plants available. Please add a plant first.")

    with tabs[4]:
        st.subheader("Audit Log")
        audit_entries = session.query(AuditLog).all()
        if audit_entries:
            df = pd.DataFrame([a.__dict__ for a in audit_entries])
            df.drop('_sa_instance_state', axis=1, inplace=True)
            st.dataframe(df)
            log_action('View Audit Log', username, "Viewed audit log.")
        else:
            st.info("No audit logs available.")

# Run the application
if __name__ == '__main__':
    main()
