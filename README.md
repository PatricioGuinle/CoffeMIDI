**MIDI Dataset Generator, Analysis Tool, and CoffeMIDI Web Application**

This project is a comprehensive tool for generating, processing, and analyzing datasets based on MIDI files. It also includes a music content recommendation system called CoffeMIDI. The main goal is to extract statistical information and musical features from MIDI files, store them in a structured format, and use that data to find similarities between songs, group them, and perform detailed musical analysis.

---

### **Motivation**

The aim of this project is to provide a toolset that can analyze MIDI files comprehensively and enable music recommendations through CoffeMIDI. The focus is on extracting detailed musical characteristics and using them to explore and recommend music interactively, offering a personalized user experience.

### **Key Features**

#### **MIDI File Feature Extraction**
- Extracts a wide range of musical features, including notes, durations, tempo, key, time signatures, and control events (such as pedal changes). It also categorizes instruments based on the General MIDI specification.
- Functions like `get_theme_df` and `get_theme_stats` create a dataframe for each MIDI file, capturing notes, chords, tempo, and event quantization.

#### **Preprocessing and Standardization**
- Uses `StandardScaler` from scikit-learn to standardize the extracted features for consistent analysis.
- Removes outliers to ensure the dataset is clean and suitable for analysis.

#### **Dataset Generation**
- The extracted features are saved in a CSV file (`df_scaled.csv`) for further analysis.
- Models and pickles are generated for categorizing variables into different groups, such as 'harmony,' 'rhythm,' and 'duration.'

#### **Categorization and Clustering**
- Clustering models categorize MIDI files, enabling the discovery of musical patterns within different groups of songs.
- Users can upload new MIDI files to categorize and add them to existing clusters.

### **CoffeMIDI: Content-Based Music Recommendation System**

CoffeMIDI builds upon the analysis capabilities of this project to create a content-based music recommendation system, allowing users to explore and discover new music interactively.

**Website**: [http://coffemidi.com.ar/](http://coffemidi.com.ar/)

#### **Dataset Information**
CoffeMIDI contains a rich dataset built from nearly **90,000 MIDI files**, with **over 500 parameters** extracted from each. The features include:
- **Instrumentation**: Categorization of instruments used.
- **Drums Rhythm**: Analysis of drum patterns.
- **Instruments Rhythm**: Rhythmic patterns of melodic instruments.
- **Tonal Range**: Span of notes used in each track.
- **Dynamics**: Changes in note velocities representing volume variation.
- **Notes Duration**: Duration of each note.
- **Theme Duration**: Total length of the piece.
- **Simultaneous Notes**: Degree of polyphony (number of notes played simultaneously).
- **Tempo**: Beats per minute (BPM) and tempo changes.
- **Harmony**: Analysis of chord usage and harmonic content.

---

### **How to Use CoffeMIDI**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PatricioGuinle/CoffeMIDI.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Dataset** (Optional):
   - Replace the example MIDI files in the `FULL_MIDI` path with your own dataset and run the dataset generator.

4. **Run the Application**:
   ```bash
   python app.py
   ```
5. **Open the Interface**:
   - Open `Front/front.html` in your browser to start using the web application.

6. **Start Exploring**:
   - Use the web app to search for songs, adjust musical parameters, and explore personalized recommendations.

### **Web Application Features**

The CoffeMIDI web application is designed to help users search, explore, and discover music based on detailed musical features.

#### **Step-by-Step Usage with Visual Guide**

1. **Search for Songs by Artist, Genre, or Theme**:
   - Use the search box to find songs by **Artist**, **Music Genre**, or **Theme name**.
   
<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step1.png?raw=true" alt="Coffe MIDI Step1"/>
</p>

2. **Select a Song from the Search Results**:
   - Choose a song from the list of search results displayed.

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 2.png?raw=true" alt="Coffe MIDI Step2"/>
</p>

3. **Adjust Musical Parameters to Explore Recommendations**:
   - Adjust various **musical parameters** such as instrumentation, rhythm, harmony, dynamics, etc., to see how the recommendations adapt. Parameters include drums rhythm, tonal range, theme duration, simultaneous notes, and more.

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 3.png?raw=true" alt="Coffe MIDI Step3"/>
</p>

   - The **Cosine Similarity** value between the selected song and similar tracks in the database is displayed in the second column. You can also click on the **Search** button to change the current theme.

4. **Listen to Recommendations**:
   - Click **Play** to listen to the recommended tracks and evaluate the quality of suggestions.

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 4.png?raw=true" alt="Coffe MIDI Step3"/>
</p>

   - You can also view a **piano roll** to see the MIDI notes visually while the song is playing.

### **Technologies Used**

- **Python**: Core programming language.
- **Flask**: REST API for searching and recommending songs.
- **Pandas, NumPy, Scikit-Learn**: Data manipulation, preprocessing, and feature scaling.
- **Mido**: For MIDI file parsing and processing.
- **Plotly**: Visualizations for understanding data.
- **HTML/CSS/JavaScript**: Web interface.

### **Applications**

- **Music Analysis**: Detailed MIDI analysis offering insights into composition, patterns, and structure.
- **Music Recommendation**: A content-based system to help users find new music based on specific features.
- **Education**: Assists students and musicians in understanding components like **harmony**, **rhythm**, and **dynamics**.

### **Flask API for Song Queries and Suggestions**

- The `app.py` file provides a RESTful API to interact with the dataset:
  - **Song Search**: `/search_titles` endpoint for searching songs by title.
  - **Similarity-Based Suggestions**: `/related_songs` endpoint uses cosine similarity to find similar songs based on their musical features.
