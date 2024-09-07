import tkinter as tk
from tkinter import filedialog, messagebox
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
from processing import process_Data

# Initialize GUI
class SpamClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Classifier")
        self.root.geometry("500x500")
        self.root.configure(bg='#89CFF0')  


        # Load models
        self.lstm_model = load_model('lstm_spam_detection_model3.h5')
        self.bilstm_model = load_model('bilstm_spam_detection_model.h5')

        
        self.create_widgets()

    def create_widgets(self):
        
        self.select_file_button = tk.Button(self.root, text="Select Text File", command=self.select_file)
        self.select_file_button.pack(pady=20)

        
        self.file_content_text = tk.Text(self.root, height=10, width=60)
        self.file_content_text.pack(pady=10)

      
        self.model_var = tk.StringVar(value="LSTM")
        self.lstm_radio = tk.Radiobutton(self.root, text="LSTM", variable=self.model_var, value="LSTM")
        self.bilstm_radio = tk.Radiobutton(self.root, text="BiLSTM", variable=self.model_var, value="BiLSTM")
        self.lstm_radio.pack()
        self.bilstm_radio.pack()

      
        self.check_button = tk.Button(self.root, text="Check", command=self.check_spam)
        self.check_button.pack(pady=20)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.file_content_text.delete(1.0, tk.END)
                self.file_content_text.insert(tk.END, content)



    def check_spam(self):
        text = self.file_content_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to classify")
            return

        tokens = process_Data(text)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tokens)
        sequences = tokenizer.texts_to_sequences([tokens])
        max_length = 1000  
        padded_sequences = pad_sequences(sequences, maxlen=max_length)

        model_choice = self.model_var.get()
        if model_choice == "LSTM":
            model = self.lstm_model
        elif model_choice == "BiLSTM":
            model = self.bilstm_model
        else:
            messagebox.showwarning("Warning", "Select a model")
            return

        prediction = model.predict(padded_sequences)
        if prediction[0][0] > 0.5:
            result = "Spam"
        else:
            result = "Ham"

        messagebox.showinfo("Result", f"The text is classified as: {result}")


root = tk.Tk()
app = SpamClassifierApp(root)
root.mainloop()
