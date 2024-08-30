import os
import time
import uuid
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from gan_process import generate_image_based_on_samples

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define folders
UPLOADS_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/dataset'
RESULT_FOLDER = 'static/hasil'
GENERATED_FOLDER = 'static/generated'

app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Function to list images in a specific folder
def list_images_in_folder(folder_path):
    return [filename for filename in os.listdir(folder_path) if filename.endswith(('jpg', 'jpeg', 'png', 'gif'))]

@app.route('/')
def index():
    folder = request.args.get('folder', '-----')
    images_folder = None
    dataset_subfolders = []
    
    if folder == 'uploads':
        images_folder = UPLOADS_FOLDER
    elif folder == 'dataset':
        images_folder = DATASET_FOLDER
        dataset_subfolders = next(os.walk(DATASET_FOLDER))[1]  # Get list of subfolders in dataset
    elif folder == 'dataset/batik-bali':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-bali')
    elif folder == 'dataset/batik-betawi':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-betawi')
    elif folder == 'dataset/batik-cendrawasih':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-cendrawasih')
    elif folder == 'dataset/batik-ciamis':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-ciamis')
    elif folder == 'dataset/batik-garutan':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-garutan')
    elif folder == 'dataset/batik-gentongan':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-gentongan')
    elif folder == 'dataset/batik-ikat-celup':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-ikat-celup')
    elif folder == 'dataset/batik-insang':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-insang')
    elif folder == 'dataset/batik-kawung':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-kawung')
    elif folder == 'dataset/batik-keraton':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-keraton')
    elif folder == 'dataset/batik-lasem':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-lasem')
    elif folder == 'dataset/batik-megamendung':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-megamendung')
    elif folder == 'dataset/batik-parang':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-parang')
    elif folder == 'dataset/batik-pekalongan':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-pekalongan')
    elif folder == 'dataset/batik-sekar':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-sekar')
    elif folder == 'dataset/batik-semarangan':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-semarangan')
    elif folder == 'dataset/batik-sidoluhur':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-sidoluhur')
    elif folder == 'dataset/batik-tambal':
        images_folder = os.path.join(DATASET_FOLDER, 'batik-tambal')
    else:
        images_folder = 'static/-----'

    images = os.listdir(images_folder)
    # Pagination
    page = request.args.get('page', 1, type=int)
    images_per_page = 30
    total_pages = -(-len(images) // images_per_page)  # Round up
    start_index = (page - 1) * images_per_page
    end_index = start_index + images_per_page
    images = images[start_index:end_index]

    return render_template('index.html', images=images, folder=folder, 
                           total_pages=total_pages, current_page=page, generated_images=[])

@app.route('/upload', methods=['POST'])
def upload():
    if 'files[]' not in request.files:
        return 'No file part'
    
    files = request.files.getlist('files[]')

    for file in files:
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOADS_FOLDER'], filename))
    
    return redirect(url_for('index'))

@app.route('/generate', methods=['POST'])
def generate():
    folder = request.form.get('folder', '-----')
    model_type = request.form.get('model', 'GAN')
    selected_images = request.form.getlist('image_paths')
    if not selected_images:
        flash('Please select at least one image.')
        return redirect(url_for('index', folder=folder))

    # Handle conversion of num_images to an integer safely
    try:
        num_images = int(request.form.get('num_images', 1))
    except ValueError:
        num_images = 1

    # Set the number of classes for CGAN
    n_classes = 10 if model_type == 'CGAN' else 1

    # Generate images and get their paths
    generated_image_paths = generate_image_based_on_samples(selected_images, num_images, model_type, n_classes)

    # Save generated images to RESULT_FOLDER with unique names
    generated_images = []
    for path in generated_image_paths:
        filename = os.path.basename(path)
        unique_filename = f"{model_type}_{uuid.uuid4().hex}.png"  # Format nama file unik
        result_path = os.path.join(RESULT_FOLDER, unique_filename)
        os.rename(path, result_path)
        generated_images.append(unique_filename)

    return render_template('index.html', images=[], selected_images=selected_images, folder=folder,
                           total_pages=1, current_page=1, generated_images=generated_images, 
                           num_images=num_images, model=model_type)

@app.route('/hasil/<filename>')
def hasil(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
