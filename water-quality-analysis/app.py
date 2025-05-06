import warnings
# Filter out specific scikit-learn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
from flask_mail import Mail, Message
import random
import os

app = Flask(__name__)

# Static file configuration
app.config['STATIC_FOLDER'] = 'static'

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ayinalakoteswararao@gmail.com'
app.config['MAIL_PASSWORD'] = 'kmmq qywm anry tsre'
app.config['MAIL_DEFAULT_SENDER'] = 'ayinalakoteswararao@gmail.com'

# Database configuration
db_config = {
    'host': 'YourUsername.mysql.pythonanywhere-services.com',  # Replace YourUsername with your PythonAnywhere username
    'user': 'YourUsername',         # Replace with your MySQL username
    'password': 'YourPassword',     # Replace with your MySQL password
    'database': 'YourUsername$water_quality'  # Replace YourUsername with your PythonAnywhere username
}

mail = Mail(app)

# Sample data for the about page
members = [
    {"name": "Member 1", "role": "Role 1"}
]

blog_content = [
    {
        "icon": "fa-tint",
        "title": "Water Quality Basics",
        "content": "Understanding water quality is crucial for public health and environmental protection. Clean water is essential for drinking, agriculture, and ecosystem balance."
    },
    {
        "icon": "fa-flask",
        "title": "Testing Methods",
        "content": "Modern water quality testing involves various parameters including pH levels, dissolved oxygen, turbidity, and chemical composition analysis."
    },
    {
        "icon": "fa-leaf",
        "title": "Environmental Impact",
        "content": "Water quality directly affects ecosystems, marine life, and biodiversity. Maintaining clean water sources is vital for environmental sustainability."
    },
    {
        "icon": "fa-home",
        "title": "Household Water Safety",
        "content": "Learn about maintaining water quality in your home, including filtration systems, regular testing, and best practices for water storage."
    },
    {
        "icon": "fa-microscope",
        "title": "Advanced Analysis",
        "content": "Discover the latest technologies and methods used in water quality analysis, from molecular testing to real-time monitoring systems."
    },
    {
        "icon": "fa-industry",
        "title": "Industrial Management",
        "content": "Explore how industries maintain water quality standards and implement sustainable water management practices."
    },
    {
        "icon": "fa-shower",
        "title": "Daily Water Usage",
        "content": "Tips and insights on how to optimize your daily water usage while maintaining quality and safety standards."
    },
    {
        "icon": "fa-filter",
        "title": "Filtration Systems",
        "content": "Compare different water filtration systems and learn which one best suits your needs for clean, safe water."
    },
    {
        "icon": "fa-seedling",
        "title": "Agricultural Impact",
        "content": "How water quality affects crop growth, soil health, and sustainable farming practices in modern agriculture."
    },
    {
        "icon": "fa-fish",
        "title": "Aquatic Ecosystems",
        "content": "Understanding the delicate balance of aquatic ecosystems and how water quality influences marine life."
    },
    {
        "icon": "fa-cloud-rain",
        "title": "Rainwater Harvesting",
        "content": "Learn about collecting and storing rainwater safely for various purposes while maintaining quality standards."
    },
    {
        "icon": "fa-temperature-high",
        "title": "Temperature Effects",
        "content": "How temperature changes affect water quality and the importance of monitoring thermal pollution."
    },
    {
        "icon": "fa-city",
        "title": "Urban Water Systems",
        "content": "Exploring city water management, treatment facilities, and distribution networks for clean water."
    },
    {
        "icon": "fa-hand-holding-water",
        "title": "Conservation Tips",
        "content": "Practical ways to conserve water in daily life while maintaining its quality for future generations."
    },
    {
        "icon": "fa-vial",
        "title": "Chemical Analysis",
        "content": "Understanding the chemical parameters that determine water quality and their significance."
    },
    {
        "icon": "fa-bacteria",
        "title": "Microbial Safety",
        "content": "Learn about waterborne pathogens, testing methods, and ensuring microbiological safety of water."
    }
]

# Load and preprocess the dataset
def load_and_preprocess_data():
    data = pd.read_csv('water_potability.csv')
    data.dropna(inplace=True)
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    # Convert data to float32 and handle missing values
    X = X.astype('float32')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    
    return X, y

# Train and save the model
def train_and_save_model(X_train, y_train):
    # Create a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Save the model
    model_path = "voting_classifier_model.pkl"
    joblib.dump(rf_model, model_path)
    print("Model trained and saved successfully")
    return rf_model

# Initialize data and model
X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save the model
model_path = 'voting_classifier_model.pkl'

def train_and_save_model():
    # Sample training data (you should replace this with your actual dataset)
    X_train = np.array([
        [7.2, 150, 500, 6, 250, 400, 12, 50, 3],  # Potable water example
        [9.5, 300, 1500, 15, 400, 800, 25, 100, 8],  # Non-potable water example
        # Add more training examples...
    ])
    y_train = np.array([1, 0])  # 1 for potable, 0 for non-potable
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, model_path)
    return model

# Load or train the model at startup
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = train_and_save_model()

# Database connection function
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

# Initialize database and create tables
def init_db():
    try:
        connection = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        cursor.execute(f"USE {db_config['database']}")
        
        # Create water quality predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS water_quality_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ph FLOAT,
                hardness FLOAT,
                solids FLOAT,
                chloramines FLOAT,
                sulfate FLOAT,
                conductivity FLOAT,
                organic_carbon FLOAT,
                trihalomethanes FLOAT,
                turbidity FLOAT,
                prediction BOOLEAN,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        connection.commit()
        print("Database initialized successfully")
    except mysql.connector.Error as err:
        print(f"Error initializing database: {err}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

# Initialize database when app starts
init_db()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            ph = float(request.form['ph'])
            hardness = float(request.form['hardness'])
            solids = float(request.form['solids'])
            chloramines = float(request.form['chloramines'])
            sulfate = float(request.form['sulfate'])
            conductivity = float(request.form['conductivity'])
            organic_carbon = float(request.form['organic_carbon'])
            trihalomethanes = float(request.form['trihalomethanes'])
            turbidity = float(request.form['turbidity'])
            
            # Validate input ranges
            if not (0 <= ph <= 14):
                return render_template('index.html', error="pH must be between 0 and 14")
            
            if any(x < 0 for x in [hardness, solids, chloramines, sulfate, conductivity, 
                                  organic_carbon, trihalomethanes, turbidity]):
                return render_template('index.html', error="All values must be positive")
            
            # Make prediction
            features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, 
                                turbidity]])
            
            # Ensure model is loaded
            global model
            if model is None:
                model = train_and_save_model()
                
            prediction = model.predict(features)[0]
            
            # Store prediction in database
            connection = get_db_connection()
            if connection:
                cursor = connection.cursor()
                query = """
                    INSERT INTO water_quality_predictions 
                    (ph, hardness, solids, chloramines, sulfate, conductivity, 
                     organic_carbon, trihalomethanes, turbidity, prediction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (ph, hardness, solids, chloramines, sulfate,
                                    conductivity, organic_carbon, trihalomethanes,
                                    turbidity, bool(prediction)))
                connection.commit()
                cursor.close()
                connection.close()
            
            # Convert prediction to string for template
            result = "Potable" if prediction == 1 else "Not Potable"
            
            return render_template('result.html', 
                                 prediction=result)
            
        except ValueError as ve:
            print(f"ValueError: {ve}")  # Debug print
            return render_template('index.html', error="Please enter valid numerical values")
        except Exception as e:
            print(f"Error: {e}")  # Debug print
            return render_template('index.html', error=f"Error: {str(e)}")

@app.route('/blog')
def blog():
    # Randomly select 8 blog posts without repetition
    selected_posts = random.sample(blog_content, 8)
    return render_template('blog.html', posts=selected_posts)

@app.route('/about')
def about():
    team_member = {
        "name": "Ayinala Koteswara Rao",
        "role": "Lead Developer & Water Quality Expert",
        "image": url_for('static', filename='images/member1.jpg'),
        "description": "Lead Developer & Water Quality Expert with extensive experience in environmental engineering. Specializing in developing innovative solutions for water quality analysis and monitoring systems.",
        "education": "B.Tech in Artificial Intelligence & Machine Learning",
        "expertise": [
            "Full Stack Development",
            "Machine Learning",
        ],
        "social_links": {
            "linkedin": "https://www.linkedin.com/in/ayinala-koteswararao-711bab271/",
            "github": "https://github.com/ayinalakoteswararao",
            "email": "ayinalakoteswararao@gmail.com"
        }
    }
    return render_template('about.html', member=team_member)

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        
        # Create email content with HTML formatting
        msg_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                <h2 style="color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                    📬 New Contact Form Submission
                </h2>
                
                <div style="margin: 20px 0;">
                    <p style="margin: 10px 0;">
                        <strong>👤 Name:</strong> {name}
                    </p>
                    <p style="margin: 10px 0;">
                        <strong>📧 Email:</strong> {email}
                    </p>
                    <p style="margin: 10px 0;">
                        <strong>📱 Phone:</strong> {phone}
                    </p>
                </div>
                
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                    <h3 style="color: #2c3e50; margin-top: 0;">💬 Message:</h3>
                    <p style="white-space: pre-line;">{message}</p>
                </div>
                
                <div style="margin-top: 20px; text-align: center; color: #666; font-size: 12px;">
                    <p>This is an automated message from your Water Quality Analysis System</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        try:
            msg = Message(
                subject='New Contact Form Submission',
                recipients=['ayinalakoteswararao@gmail.com'],
                html=msg_body  # Changed from body to html to support HTML formatting
            )
            mail.send(msg)
            return jsonify({'success': True, 'message': 'Message sent successfully!'})
        except Exception as e:
            print(f"Error sending email: {e}")
            return jsonify({'success': False, 'message': 'Failed to send message. Please try again later.'}), 500
            
    return render_template('contact.html')

if __name__ == '__main__':
    app.run()
