# PlantCare AI – Real-time Bilingual Plant Disease Assistant

A Flask web application with live camera capture, MobileNetV2 transfer-learning classification, disease information, and English/Telugu text + browser voice interaction.

## Important: model and dataset are not bundled

This package contains the complete application code and training pipeline. It does **not** contain a trained disease model or a leaf dataset. A camera cannot identify diseases by itself: the classifier must first be trained using labelled images. The app intentionally reports that model setup is required until you train and save the model.

For a meaningful result, collect your own real-field images and obtain labels verified by an agricultural expert. You may use a pretrained ImageNet MobileNetV2 backbone for transfer learning; this is not the same as using an existing plant-disease dataset. Include healthy examples and each disease class you intend to support.

## 1. Install through Anaconda Navigator (Windows)

1. Install Anaconda or Miniconda.
2. Open **Anaconda Navigator → Environments → Import** and select `environment.yml`, or open Anaconda Prompt and run:
   ```bash
   conda env create -f environment.yml
   conda activate plantcare-ai
   ```
3. In Anaconda Navigator, select the `plantcare-ai` environment and open a terminal, or use Anaconda Prompt.
4. From this project folder run:
   ```bash
   pip install -r requirements.txt
   ```

TensorFlow 2.15.1 is pinned for Python 3.10 compatibility. On Windows, this setup is intended for CPU training/inference. GPU support on native Windows is more complicated; use WSL2 if GPU acceleration is required.

## 2. Prepare your own dataset

Create one folder per class. The folder name is the exact class label:

```text
PlantCare_Bilingual_AI/
  dataset/
    Tomato___healthy/
      leaf_001.jpg
      leaf_002.jpg
    Tomato___Early_blight/
      leaf_001.jpg
    Tomato___Late_blight/
      leaf_001.jpg
    Potato___healthy/
      leaf_001.jpg
    Potato___Early_blight/
      leaf_001.jpg
```

- Use clear, in-focus images with one main leaf.
- Collect many varied examples per class (ideally hundreds, not just a handful).
- Include different farms, phones, lighting, backgrounds, growth stages and disease severity.
- Avoid near-duplicate images across training and validation.
- Keep a separate test set of real field images that is never used during training.
- Confirm labels with a plant pathologist/agricultural expert where possible.
- Begin with a small, manageable crop/disease scope.

## 3. Train the model

From the project root, with `plantcare-ai` activated:

```bash
python training/train_model.py
```

The script:
- Reads class-labelled folders from `dataset/`.
- Creates a training/validation split.
- Uses ImageNet-pretrained MobileNetV2 as a transfer-learning backbone.
- Adds image augmentation and a classification head.
- Fine-tunes the last backbone layers.
- Saves:
  - `models/plant_disease_mobilenetv2.keras`
  - `models/class_names.json`

Do not present validation accuracy as real-world accuracy. Evaluate using a separate field-image test set and tune the confidence threshold based on those results.

## 4. Start the application

```bash
python app.py
```

Open Chrome and visit:

`http://127.0.0.1:5000`

Click **Start camera**, allow camera permission, position a single leaf in good light, then click **Capture & predict**.

Camera access works on `localhost` / `127.0.0.1` or an HTTPS deployment. It will not work from an ordinary insecure remote HTTP address.

## 5. English and Telugu voice

- Select English or Telugu from the language dropdown.
- Type a question or press **Speak** to use browser speech recognition.
- The assistant response is displayed as text and spoken using browser speech synthesis.
- Speech recognition and Telugu voice availability depend on the browser, operating system and installed speech services. Chrome support and installed Telugu voice packs vary by device.
- If a Telugu voice is unavailable, install/enable a Telugu voice in the operating system or integrate a cloud Speech-to-Text / Text-to-Speech provider.

## 6. Enable open-ended AI conversation (optional)

Copy `.env.example` to `.env` and set your provider details:

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=
LLM_MODEL=gpt-4o-mini
CONFIDENCE_THRESHOLD=0.65
```

For an OpenAI-compatible provider such as Groq, set its documented base URL and model name. Keep API keys private; never place them in JavaScript or commit `.env` to source control.

Without an LLM key, the app uses a limited safety fallback. The fallback is not a full conversational AI.

## 7. Current limitations / safety

- The included disease information file contains only example entries for a few tomato and potato classes. Extend it for every class you train.
- The prediction endpoint uses the model's maximum softmax probability as a confidence-like score. Neural-network softmax values can be overconfident; calibrate and validate before relying on them.
- A low score produces an Unknown / Uncertain result, but this is not a complete out-of-distribution detector.
- Similar-looking diseases, nutrient deficiencies, pests and environmental stress can be confused.
- This is an educational prototype, not a certified diagnostic system. Do not recommend pesticide products or dosages solely from a model prediction. Confirm treatment with local agricultural experts.

## Troubleshooting

**Model files missing:** Prepare `dataset/` and run `python training/train_model.py`.

**Camera permission denied:** Use Chrome at `http://127.0.0.1:5000`, allow camera permission, and close other apps using the camera.

**TensorFlow install error:** Confirm Python 3.10 and 64-bit environment:
```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Telugu speech not recognized / spoken:** Browser speech services and local language voice packs vary. Test the browser's speech support on the target device or configure cloud speech services.

**LLM chat not working:** Check `.env`, provider base URL, model identifier, API quota and internet connection.


## Camera troubleshooting (Windows / Chrome)
1. Start Flask with `python app.py` and open exactly `http://127.0.0.1:5000` in Chrome.
2. Click **Start camera** and choose **Allow** in the browser permission prompt.
3. If the preview stays black, open Windows **Settings → Privacy & security → Camera** and enable Camera access and desktop-app access. Close Camera, Teams, Zoom, or other apps that may be using the camera, then refresh the page.
4. The camera preview should show a live image before you click **Capture & predict**.

## Configure the chatbot with Groq
1. Copy `.env.example` and rename the copy to `.env` in the same folder as `app.py`.
2. Create a Groq API key at `https://console.groq.com/keys`.
3. Open `.env` and replace `PASTE_YOUR_GROQ_API_KEY_HERE` with your key. Keep the key private.
4. Save the file, stop Flask with `Ctrl+C`, then restart using `python app.py`.
5. Select English or Telugu and send a test question. The backend uses Groq's OpenAI-compatible endpoint and model setting shown in `.env`. If the model is no longer available, select an active chat model from Groq's models page and update `LLM_MODEL`.

The chatbot requires internet access and an active API account / available quota. Browser speech recognition and speech playback are separate features and depend on Chrome and installed language voices.
