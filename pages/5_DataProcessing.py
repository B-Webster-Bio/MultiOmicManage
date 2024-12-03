import streamlit as st
import nbformat
from nbconvert import HTMLExporter
from IPython.display import HTML

def display_notebook(notebook_path):
    """
    Render a Jupyter notebook as HTML inside a Streamlit app
    
    Args:
        notebook_path (str): Path to the .ipynb notebook file
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert notebook to HTML
    html_exporter = HTMLExporter()
    html_output, _ = html_exporter.from_notebook_node(nb)
    
    # Display the HTML in Streamlit
    st.components.v1.html(html_output, height=800, scrolling=True)

def main():
    st.title('Jupyter Notebook Viewer')
    
    # Allow file upload of notebook
    uploaded_file = st.file_uploader("Choose a Jupyter Notebook", type="ipynb")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_notebook.ipynb", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the notebook
        display_notebook("temp_notebook.ipynb")

if __name__ == "__main__":
    main()