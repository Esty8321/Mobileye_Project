# 🌸 Flower Classifier UI

This is the **frontend (UI)** of the Flower Classifier project.  
It allows users to:

- Upload an image of a flower 🌼  
- Send it to the backend server for prediction 🤖  
- Display the predicted class and top-3 probabilities 📊  
- Annotate an image with a custom tag 🏷️ (the image + label are saved for future training)

---

## ⚙️ Setup

1. **Install dependencies**

Make sure you have [Node.js](https://nodejs.org/) (>= 18) installed.  
Then, inside the `ui/` folder, run:

```bash
 npm install

2.  Start the development server

3. npm run dev
-This will start the Vite dev server.

Usage

Open your browser and go to:
http://localhost:5173/

Upload an image (JPG/PNG).

Click Predict → the UI will display the predicted class.

Optionally, enter a label and click Annotate → the image + label are sent to the backend and saved.