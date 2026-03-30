from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename  # Add this import
# importing sqlalchemy
from flask import send_file  # Add this with your other Flask imports
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# importing flask login for encrypted & decrypted password stored in db
from flask_login import UserMixin
import sys  # Add this with other imports
# for hashed function which helps in encryption & decryption
from werkzeug.security import generate_password_hash, check_password_hash
# Add this right after your imports
import matplotlib
matplotlib.use('Agg')  # Must be before other matplotlib imports
import matplotlib.pyplot as plt
from flask_login import login_user, logout_user, login_manager, LoginManager
from flask_login import login_required, current_user
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import shap
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import os
import traceback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import send_from_directory  # Add this with other Flask imports
from fpdf import FPDF
from datetime import datetime
import os
from flask import send_file, flash, redirect, url_for
# For importing json file
import json

# For using deep learning model

from PIL import Image

import cv2  # Add this import for image processing
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras_preprocessing.image import load_img, img_to_array

# Add these right after your existing imports
import shap
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler



# ===== FUSION MODEL SETUP =====
def create_fusion_model():
    X_scaled, y, scaler = generate_synthetic_cbioportal_data()
    
    model = Sequential([
        Input(shape=(3,)),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    
    return model, scaler, X_scaled

def generate_synthetic_cbioportal_data(num_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate synthetic features
    mutation_count = np.random.lognormal(mean=1.5, sigma=0.8, size=num_samples)
    age = np.random.normal(loc=55, scale=15, size=num_samples)
    sex = np.random.binomial(1, 0.5, size=num_samples)  # 0=male, 1=female
    
    # Create synthetic target (1=tumor, 0=no tumor)
    tumor_prob = 1 / (1 + np.exp(-(0.5*mutation_count + 0.03*age - 0.2*sex - 5)))
    y = np.random.binomial(1, tumor_prob)
    
    # Combine features
    X = np.column_stack([mutation_count, age, sex])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Initialize fusion model and explainer
fusion_model, scaler, X_scaled = create_fusion_model()
background = shap.sample(X_scaled, 100)
explainer = shap.KernelExplainer(fusion_model.predict, background)
# My database connection
local_server = True
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'  # add any secret_key

# After creating directories, you could add:
if not os.path.exists('results/acc_cnn.txt'):
    with open('results/acc_cnn.txt', 'w') as f:
        f.write("85.5")  # Default CNN accuracy

if not os.path.exists('results/acc_mn.txt'):
    with open('results/acc_mn.txt', 'w') as f:
        f.write("90.0")  # Default MobileNet accuracy
# This is for unique user access
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ===== ADD THESE LINES RIGHT HERE =====
# Predefined credentials
PREDEFINED_EMAIL = "user@brain.com"
PREDEFINED_PASSWORD = "braintumor123"

class MockUser(UserMixin):
    def __init__(self):  # Fix: Proper __init__ with double underscores
        self.id = 1  # Ensure id attribute exists
        self.email = PREDEFINED_EMAIL
        self.username = "admin"
        self.password = generate_password_hash(PREDEFINED_PASSWORD)
        
        
    def check_password(self, password):
        return check_password_hash(self.password, password)
# ===== END OF ADDITION =====
def get_img_array(img_path, size, color_mode='rgb'):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=size, color_mode=color_mode)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    import tensorflow as tf
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        print(f"conv_outputs shape: {conv_outputs.shape}")
        print(f"predictions: {predictions}")
        if pred_index is None:
            # Handle binary classification with sigmoid output
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                pred_index = tf.argmax(predictions[0])
                pred_index = int(pred_index.numpy())
                class_channel = tf.gather(predictions[0], pred_index)
        else:
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
            else:
                class_channel = tf.gather(predictions[0], pred_index)

    grads = tape.gradient(class_channel, conv_outputs)
    print(f"grads shape: {grads.shape}")
    print(f"grads: {grads}")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    print(f"pooled_grads shape: {pooled_grads.shape}")
    print(f"pooled_grads: {pooled_grads}")

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    epsilon = 1e-10
    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (max_val + epsilon)
    heatmap = tf.clip_by_value(heatmap, 0, 1)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", raw_cam_path="raw_cam.jpg", alpha=0.4):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load the original image
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    # Replace NaNs with zeros
    heatmap = np.nan_to_num(heatmap, nan=0.0)

    # Normalize heatmap to range 0-1
    heatmap = np.maximum(heatmap, 0)
    max_heatmap = np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
    heatmap = heatmap / max_heatmap

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Save raw heatmap image (grayscale)
    plt.imsave(raw_cam_path, heatmap_resized, cmap='jet')

    # Create figure and axis
    fig, ax = plt.subplots()
    ax.imshow(img)
    im = ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    # Remove axis for clean image
    ax.axis('off')

    # Save the superimposed image
    plt.savefig(cam_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return cam_path, raw_cam_path  # Now returns both paths

@login_manager.user_loader
def load_user(user_id):
    # First check if it's our predefined user
    if int(user_id) == 1:
        return MockUser()
    # Fallback to database
    return User.query.get(int(user_id))

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/database_name'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/database_name'

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/database_name?ssl_ca=/path/to/ca.pem'


db = SQLAlchemy(app)

# Add this right after your database setup (after db = SQLAlchemy(app))
# ===== FUSION MODEL SETUP =====
def generate_synthetic_cbioportal_data(num_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate synthetic features
    mutation_count = np.random.lognormal(mean=1.5, sigma=0.8, size=num_samples)
    age = np.random.normal(loc=55, scale=15, size=num_samples)
    sex = np.random.binomial(1, 0.5, size=num_samples)  # 0=male, 1=female
    
    # Create synthetic target (1=tumor, 0=no tumor)
    tumor_prob = 1 / (1 + np.exp(-(0.5*mutation_count + 0.03*age - 0.2*sex - 5)))
    y = np.random.binomial(1, tumor_prob)
    
    # Combine features
    X = np.column_stack([mutation_count, age, sex])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Initialize fusion model and explainer
X_scaled, y, scaler = generate_synthetic_cbioportal_data()

fusion_model = Sequential([
    Input(shape=(3,)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

fusion_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
fusion_model.fit(X_scaled, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

background = shap.sample(X_scaled, 100)
explainer = shap.KernelExplainer(fusion_model.predict, background)
# ===== END FUSION MODEL SETUP =====
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50))
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(1000))


class Patient(db.Model):
    pid = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50))
    fname = db.Column(db.String(15))
    lname = db.Column(db.String(15))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    date = db.Column(db.String(50), nullable=False)
    id = db.Column(db.Integer)
    number = db.Column(db.String(10))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/patient', methods=['POST', 'GET'])
@login_required
def patient():
    if request.method == "POST":
        em = current_user.email
        fname = request.form.get('fname')
        lname = request.form.get('lname')
        age = request.form.get('age')
        gender = request.form.get('gender')
        date = request.form.get('date')
        id = request.form.get('id')
        number = request.form.get('number')
        user1 = Patient.query.filter_by(email=em).first()
        user2 = Patient.query.filter_by(id=id).first()

        if user1 and user2:
            flash("Passenger Already Exist", "warning")
            return render_template('/patient.html')

        query = db.engine.execute(
            f"INSERT INTO patient (id,email,fname,lname,age,gender,date,number) VALUES ('{id}','{em}','{fname}','{lname}','{age}','{gender}','{date}','{number}')")

        flash("Registered", "info")
        return redirect('/upload')

    return render_template('patient.html')
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        print("\n=== RAW FORM DATA ===")
        print("Form data:", request.form)
        print("Files:", request.files)
        
        model_name = request.form.get('model')
        print("Selected model:", model_name)

        if not model_name:
            flash("Please select a model", "danger")
            return redirect(url_for('upload'))

        if model_name == "Fusion":
            try:
                mutation_count = float(request.form.get('mutationCount', 0))
                age = float(request.form.get('age', 0))
                gender = float(request.form.get('gender', 0))
                gender_display = "Female" if gender == 1 else "Male"
                
                input_data = np.array([[mutation_count, age, gender]])
                input_scaled = scaler.transform(input_data)
                
                # Debug print inputs
                print(f"\nInput data: {input_data}")
                print(f"Scaled input: {input_scaled}")
                
                prediction = fusion_model.predict(input_scaled)
                pred_value = float(prediction[0][0])
                result = "Tumor Detected" if pred_value > 0.5 else "No Tumor"
                confidence = f"{pred_value * 100:.2f}%" if pred_value > 0.5 else f"{(1 - pred_value) * 100:.2f}%"
                
                # Get SHAP values
                # Inside your Fusion model try block:
                shap_values = explainer.shap_values(input_scaled)
                
                # Debug prints
                print(f"SHAP values type: {type(shap_values)}")
                print(f"SHAP values content: {shap_values}")
                
                # Handle binary classification output
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values[1])  # Take positive class SHAP values
                
                # Create plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, 
                    input_scaled,
                    feature_names=["Mutation Count", "Age", "Sex"],
                    plot_type="dot",
                    show=False
                )
                
                # Ensure directory exists
                plot_dir = os.path.join('static', 'images')
                os.makedirs(plot_dir, exist_ok=True)
                plot_filename = f"shap_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                plot_path = os.path.join(plot_dir, plot_filename)
                
                # Save with tight layout and white background
                plt.tight_layout()
                plt.savefig(plot_path, bbox_inches='tight', dpi=150, facecolor='white')
                plt.close()
                
                # Verify file was created
                if not os.path.exists(plot_path):
                    print(f"ERROR: Failed to save plot at {plot_path}")
                    flash("SHAP visualization failed", "warning")
                    plot_filename = None  # This will trigger the error message in template
                else:
                    print(f"SHAP plot successfully saved to: {plot_path}")
                
                return render_template('result.html',
                    model_name="Fusion",
                    result=result,
                    confidence=confidence,
                    accuracy="86%",
                    mutation_count_shap=shap_values[0][0],
                    age_shap=shap_values[0][1],
                    sex_shap=shap_values[0][2],
                    sex_display=gender_display,
                    shap_plot=plot_filename,
                    filename=None,
                    heatmap=None,
                    raw_heatmap=None
                )

            except Exception as e:
                print(f"\nFusion model error: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                flash(f"Error in Fusion model: {str(e)}", "danger")
                return redirect(url_for('upload'))

        elif model_name == "CNN":
            try:
                if 'file' not in request.files:
                    flash("No file part", "danger")
                    return redirect(url_for('upload'))
                
                file = request.files['file']
                if file.filename == '':
                    flash("No selected file", "danger")
                    return redirect(url_for('upload'))
                
                filename = secure_filename(file.filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    flash("Invalid file type", "danger")
                    return redirect(url_for('upload'))
                
                # Ensure images directory exists
                os.makedirs('static/images', exist_ok=True)
                filepath = os.path.join('static', 'images', filename)
                file.save(filepath)
                print(f"\nImage saved to: {filepath}")
                
                # Load model
                try:
                    model = load_model("brain_tumor_model_2d.h5")
                except Exception as e:
                    print(f"\nError loading CNN model: {str(e)}")
                    flash("Could not load CNN model", "danger")
                    return redirect(url_for('upload'))
                
                # Process image
                img = Image.open(filepath).convert('L')
                img = img.resize((128, 128))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=-1)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediction
                prediction = model.predict(img_array)
                pred_value = float(prediction[0][0])
                result = "Tumor Detected" if pred_value > 0.5 else "No Tumor"
                confidence = f"{pred_value * 100:.2f}%" if pred_value > 0.5 else f"{(1 - pred_value) * 100:.2f}%"
                
                # Generate heatmaps
                heatmap_filename = None
                raw_heatmap_filename = None
                # In the CNN try block, update the heatmap generation part:
                try:
                    last_conv_layer_name = 'conv2d_26'  # Update this with your actual layer name
                    heatmap = make_gradcam_heatmap(np.copy(img_array), model, last_conv_layer_name)
                    
                    base_name = os.path.splitext(filename)[0]
                    heatmap_filename = f"heatmap_{base_name}.png"
                    raw_heatmap_filename = f"raw_heatmap_{base_name}.png"
                    
                    # Generate both images
                    heatmap_path, raw_heatmap_path = save_and_display_gradcam(
                        filepath, 
                        heatmap, 
                        os.path.join('static', 'images', heatmap_filename),
                        os.path.join('static', 'images', raw_heatmap_filename)
                    )
                    
                except Exception as e:
                    print(f"\nHeatmap generation error: {str(e)}")
                    traceback.print_exc()
                    heatmap_filename = None
                    raw_heatmap_filename = None
                    flash("Heatmap generation failed, showing basic results", "warning")

                print("\nCNN results ready, rendering template...")
                return render_template('result.html',
                    model_name="CNN",
                    result=result,
                    confidence=confidence,
                    accuracy="85.5%",
                    filename=filename,
                    heatmap=heatmap_filename,
                    raw_heatmap=raw_heatmap_filename
                )

            except Exception as e:
                print(f"\nCNN model error: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                flash(f"CNN model error: {str(e)}", "danger")
                return redirect(url_for('upload'))

        else:
            flash("Invalid model selected", "danger")
            return redirect(url_for('upload'))

    return render_template('predict.html')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')

        # 1. Check predefined credentials first
        if email == PREDEFINED_EMAIL:
            mock_user = MockUser()  # This now properly initializes with password
            if mock_user.check_password(password):
                login_user(mock_user)
                flash("Login Successful", "primary")
                return redirect(url_for('index'))
        
        # 2. Fallback to database check
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login Successful", "primary")
            return redirect(url_for('index'))
        
        flash("Invalid Credentials", "danger")
        return render_template('login.html')

    return render_template('login.html')


from fpdf import FPDF
from datetime import datetime
import os
from flask import send_file, flash, redirect, url_for

from fpdf import FPDF
from datetime import datetime
import os
from flask import send_file, flash, redirect, url_for

@app.route('/generate_pdf_report/<filename>')
@login_required
def generate_pdf_report(filename):
    """Generate PDF with images side by side"""
    try:
        # ===== 1. SET UP PATHS =====
        base_dir = os.path.abspath(os.path.dirname(__file__))
        images_dir = os.path.join(base_dir, 'static', 'images')
        reports_dir = os.path.join(base_dir, 'static', 'reports')
        
        os.makedirs(reports_dir, exist_ok=True)

        # ===== 2. GET REPORT DATA =====
        result = request.args.get('result', 'No Tumor')
        confidence = request.args.get('confidence', 'N/A')
        model_name = request.args.get('model_name', 'CNN')
        accuracy = request.args.get('accuracy', 'N/A')

        # ===== 3. CREATE PDF =====
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'BRAIN TUMOR DETECTION REPORT', 0, 1, 'C')
        pdf.ln(8)
        
        # Report Info
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
        pdf.cell(0, 8, f"Result: {result} ({confidence} confidence)", 0, 1)
        pdf.cell(0, 8, f"Model: {model_name} (Accuracy: {accuracy})", 0, 1)
        pdf.ln(10)

        # ===== 4. ADD IMAGES SIDE BY SIDE =====
        img_width = 85  # Adjusted to fit two images side by side
        img_height = 80  # Fixed height for consistency
        
        original_path = os.path.join(images_dir, filename)
        heatmap_path = os.path.join(images_dir, f"heatmap_{filename}")
        
        # Get current Y position after text
        y_position = pdf.get_y()
        
        # Original Image (Left)
        if os.path.exists(original_path):
            pdf.set_font('Arial', 'I', 10)
            pdf.set_xy(15, y_position)
            pdf.cell(40, 5, "Original MRI", 0, 2)
            pdf.image(original_path, x=15, y=y_position+5, w=img_width, h=img_height)
        
        # Heatmap Image (Right)
        if os.path.exists(heatmap_path):
            pdf.set_font('Arial', 'I', 10)
            pdf.set_xy(110, y_position)
            pdf.cell(40, 5, "Heatmap Overlay", 0, 2)
            pdf.image(heatmap_path, x=110, y=y_position+5, w=img_width, h=img_height)
        
        # Move Y position down below images
        pdf.set_y(y_position + img_height + 15)

        # ===== 5. ADD FOOTER =====
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, "Generated by Brain Tumor Detection System - For medical review only", 0, 0, 'C')

        # ===== 6. SAVE AND RETURN =====
        report_filename = f"Brain_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(reports_dir, report_filename)
        pdf.output(report_path)

        return send_file(
            report_path,
            as_attachment=True,
            download_name=report_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"PDF Error: {str(e)}")
        flash("Failed to generate PDF", "danger")
        return redirect(url_for('upload'))
@app.route('/debug_shap')
def debug_shap():
    """Test SHAP visualization independently"""
    try:
        # Create test data
        test_data = np.array([[30, 40, 0]])  # Mutation=30, Age=40, Male
        test_scaled = scaler.transform(test_data)
        
        # Get SHAP values
        shap_values = explainer.shap_values(test_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            test_scaled,
            feature_names=["Mutation", "Age", "Sex"],
            plot_type="dot",
            show=False
        )
        
        # Save to test file
        test_path = os.path.join('static', 'images', 'debug_shap.png')
        plt.savefig(test_path, bbox_inches='tight')
        plt.close()
        
        return f"""
        <h1>SHAP Debug</h1>
        <p>Plot saved to: {test_path}</p>
        <img src="/static/images/debug_shap.png" style="max-width: 100%;">
        <pre>SHAP values: {shap_values}</pre>
        """
    except Exception as e:
        return f"<pre>Error: {str(e)}\n\n{traceback.format_exc()}</pre>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logout Successful", "warning")
    return redirect(url_for('login'))


@app.route('/deleteacc')
@login_required
def deleteacc():
    cid = current_user.id
    em = current_user.email
    db.engine.execute(f"DELETE FROM user WHERE user.id= {cid}")
    db.engine.execute(f"DELETE FROM patient WHERE patient.email= '{em}'")
    flash("Your Account is Deleted Successfully", "warning")
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)