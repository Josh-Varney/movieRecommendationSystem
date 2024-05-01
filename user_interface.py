import sys
import random
import content_cosine_based as cosine
import content_euclidean_tag_based as euclidean
import cosine_euclidean_tag_based as cosine_euclidean
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox, QListWidget, QListWidgetItem, QCheckBox
from PyQt5.QtCore import Qt

class SearchWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Search Interface")
        self.setGeometry(200, 200, 400, 350)  

        # widget creation
        self.label = QLabel("Movie Search Query:")
        self.label.setStyleSheet("font-size: 16px;")  
        self.search_box = QLineEdit()
        self.search_box.setStyleSheet("font-size: 16px;")  
        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("font-size: 16px; padding: 10px;")  
        self.movie_list = QListWidget()  # widget list of movies
        self.checkbox_cosine = QCheckBox("Use Cosine")
        self.checkbox_cosine.setChecked(True)  # default selection is cosine

        # screen layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.search_box)
        layout.addWidget(self.checkbox_cosine)  
        layout.addWidget(self.submit_button)
        layout.addWidget(self.movie_list) 

        self.setLayout(layout)

        # button functionality
        self.submit_button.clicked.connect(self.submit_search)

    def submit_search(self):
        # get text from search box
        query = self.search_box.text()

        # Validate the input
        if not query:
            # error if empty
            self.show_error_message("Search query cannot be empty.")
        elif len(query) > 255:
            # error too long
            self.show_error_message("Search query is too long. Please keep it under 255 characters.")
        else:
            # selected movie recommedation method
            if self.checkbox_cosine.isChecked():
                movie_list = self.filter_query_cosine(query)
            else:
                movie_list = self.filter_query_tag(query)
            
            self.display_movies(movie_list)

    def filter_query_cosine(self, query):
        """
        Filter the search query using cosine similarity.
        
        Args:
            query (str): The search query entered by the user.
        
        Returns:
            list: The list of filtered movies.
        """
        # this will take some time
        return cosine.cosineRecommendation(query)

    def filter_query_tag(self, query):
        """
        Filter the search query using tag-based recommendation.
        
        Args:
            query (str): The search query entered by the user.
        
        Returns:
            list: The list of filtered movies.
        """
        # this will take some time
        return euclidean.tagRecommendation(query)
    
    def filter_by_both(self, query):
        """
        Filter the search query using tag-based euclidean and cosine recommendation.
        
        Args:
            query (str): The search query entered by the user.
        
        Returns:
            list: The list of filtered movies.
        """
        pass
        
    def display_movies(self, movie_list):
        """
        Display the list of movies on the screen.
        
        Args:
            movie_list (list): The list of movies to display.
        """
        # Clear existing items from the movie list widget
        self.movie_list.clear()
        
        if not movie_list:
            # If no movies are recommended, display a message box
            self.show_info_message("No movies recommended.")
        else:
            # Add each movie to the movie list widget
            for movie in movie_list:
                item = QListWidgetItem(movie)
                self.movie_list.addItem(item)

    def show_info_message(self, message):
        """
        Display an information message dialog box.
        
        Args:
            message (str): The information message to be displayed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Information")
        msg_box.setText(message)
        msg_box.exec_()


    def show_error_message(self, message):
        """
        Display an error message dialog box.
        
        Args:
            message (str): The error message to be displayed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchWidget()
    window.show()
    sys.exit(app.exec_())
