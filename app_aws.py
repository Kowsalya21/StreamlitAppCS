import streamlit as st
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import io
import base64
import numpy as np

# ***************************************  METHOD 1 - CLASSIFICATION - AI/REAL *************************************** 

def classify_image(uploaded_file):
    model_name = "openai/clip-vit-large-patch14-336"
    classifier = pipeline("zero-shot-image-classification", model=model_name)
    labels_for_classification = ["AI generated damaged car image", "Real damaged car image"]

    # Read the uploaded image file as a PIL image
    image = Image.open(uploaded_file)

    # Convert the image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Perform classification
    scores = classifier(image_base64, candidate_labels=labels_for_classification)
    return scores[0]['label'], scores[0]['score']

# ***************************************  METHOD 1 - end  *************************************** 

# ***************************************  METHOD 3 - DETECTION : TYPE OF DAMAGE *************************************** 

if 'class_dict' not in st.session_state:
    st.session_state.class_dict = {}
if 'result_labels' not in st.session_state:
    st.session_state.result_labels = []

def detect_damage(uploaded_file):
    model_name = "openai/clip-vit-large-patch14-336"
    classifier = pipeline("zero-shot-image-classification", model=model_name)

    # Define labels for different types of damages
    damage_labels = [
        "Dents and Dings",
        "Paint Damage",
        "Scratches",
        "Rear Bumper damage",
        "Front Bumper damage",
        "Rear Fender damage",
        "Front fender damage",
        "Windshield Cracks",
        "Headlight Damage",
        "Taillight Damage",
        "Mirror Damage",
        "Grille Damage",
        "Roof Damage",
        "Door damage"
    ]

    # Read the uploaded image file as a PIL image
    image = Image.open(uploaded_file)

    # Convert the image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Perform classification
    scores = classifier(image_base64, candidate_labels=damage_labels, max_length=5)
    
    # Extract the top 5 labels and scores
    top_labels = [score['label'] for score in scores[:5]]
    top_scores = [score['score'] for score in scores[:5]]

    return top_labels, top_scores


def classify_image_parts(uploaded_file):
    model_name = "openai/clip-vit-large-patch14-336"
    classifier = pipeline("zero-shot-image-classification", model=model_name)
    
    labels_for_classification = {
        "Left": ["Front Left Wheel", "Left Front Fender", "Left Front Door", "Left Rear Door", "Left Rear Wheel", "Rear Left Fender", "Door Handles", "Side Mirrors"],
        "Right": ["Front Right Wheel", "Right Front Fender", "Right Front Door", "Right Rear Door", "Right Rear Wheel", "Rear Right Fender", "Door Handles", "Side Mirrors"],
        "Front": ["Front Bumper", "Grille", "Headlights", "Hood", "Front Fenders", "Windshield", "Roof", "Badge or Emblem"],
        "Back": ["Rear Bumper", "Taillights", "Trunk or Tailgate", "Rear Windshield", "Rear Spoiler", "Exhaust Pipe", "Muffler"]
    }

    # Read the uploaded image file as a PIL image
    image = Image.open(uploaded_file)

    # Convert the image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Perform classification
    scores = classifier(image_base64, candidate_labels=sum(labels_for_classification.values(), []))  # Flatten the list of labels
    highest_label = scores[0]['label']

    # Find the category corresponding to the highest-scoring label
    category = next(key for key, value in labels_for_classification.items() if highest_label in value)

    # Use labels from the specific category
    category_labels = labels_for_classification[category]
    scores = classifier(image_base64, candidate_labels=category_labels)
    top_three_labels = [score['label'] for score in scores[:3]]

    # Return the result
    return {
        "Category": category,
        "Top 3 Labels": top_three_labels
    }

def check_legitimacy(claim_description, damaged_parts):
    from transformers import pipeline

    # Load the zero-shot classification model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Set up the sequence to classify and the candidate labels
    sequence_to_classify = claim_description
    candidate_labels = damaged_parts

    # Use the classifier to get the classification result
    result = classifier(sequence_to_classify, candidate_labels)

    # Return the result
    return result


# ***************************************  METHOD 3 - end  *************************************** 


# ***************************************  METHOD MAIN  *************************************** 

def main():
    st.markdown("<h1 style='text-align: center;position: font-size: 40px; color: grey'>Claim Legitimacy Predictor</h1>", unsafe_allow_html=True)
    st.sidebar.write("#### Upload image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'png'])
    st.sidebar.write("#### Claim description")
    claim_description = st.sidebar.text_input('Enter description here 👇', key='claim_description')

    # Display buttons only when both image and description are provided
    if uploaded_file is not None and claim_description.strip():
        image = Image.open(uploaded_file)
        st.write("##### Uploaded image")
        st.image(image, width=250)
        st.write(
            '<style>div[data-testid="stImage"] img {display: block; margin-left: auto; margin-right: auto; max-height: 250px;}</style>',
            unsafe_allow_html=True)
        st.write("")
        st.write("##### Analysis on Uploaded image")
        st.write("")
        st.sidebar.write("#### Click any button to perform analysis")

        # METHOD 1 - CLASSIFICATION - AI/REAL
        if st.sidebar.button('Classify', key='classify'):
            # Perform image classification using CLIP model
            prediction_label, prediction_score = classify_image(uploaded_file)
            st.write("###### Classification - AI-generated or Real damaged car image: ")
            st.write(f"{prediction_label} with prediction score {prediction_score:.3f}")

        # METHOD 3 -  DETECTION - PARTS and TYPE OF DAMAGE
        from fuzzywuzzy import fuzz

        if st.sidebar.button('Detect Parts & Damage', key='detect_damage_parts'):
            st.write("---------------Detecting parts ----------------")
            # Perform image classification using CLIP model -  detect parts
            st.session_state.class_dict = classify_image_parts(uploaded_file)
            st.write(f"Category - {st.session_state.class_dict['Category']}")
            st.write(f"Top 3 Labels - {st.session_state.class_dict['Top 3 Labels']}")

            st.write("----------------Detecting type of damage -------------------------")
            # Detect damage
            st.session_state.result_labels, result_scores = detect_damage(uploaded_file)
            for result_label, result_score in zip(st.session_state.result_labels, result_scores):
                st.write(f"{result_label}: {result_score}")
            st.write("------------- Common between parts and type of damage----------------------------")
            # Find common words using string similarity
            threshold = 50  # Adjust the threshold as needed
            common_words = []

            for top_label in st.session_state.class_dict['Top 3 Labels']:
                for result_label in st.session_state.result_labels:
                    similarity_ratio = fuzz.token_set_ratio(top_label, result_label)
                    if similarity_ratio >= threshold:
                        common_words.append(top_label)
                        break  # Break after finding the first match

            # Store the common words in a variable named 'damaged_parts'
            damaged_parts = ', '.join(common_words)
            # st.write(f"Common Words: {damaged_parts}")
            damaged_parts_txt = "The damaged parts of the car are " + damaged_parts 
            st.write(damaged_parts_txt)

            st.write("-----------------checking legitimacy ------------------------")

            comparison_result = check_legitimacy(claim_description, damaged_parts)
            # st.write(result)
            # Access all labels and scores
            labels = comparison_result['labels']
            scores = comparison_result['scores']

            # Print all labels and scores
            for label, score in zip(labels, scores):
                if score > 0.75:
                    st.write(f"{label}, {score}")
                    st.write("The claim is legitimate.")
                    break
            else:
                st.write("The claim is Not legitimate.")
        
if __name__ == '__main__':
    main()

# ***************************************  METHOD MAIN - end *************************************** 